from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ee, pickle, numpy as np, pandas as pd, logging
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

carbon_model = None

@app.on_event("startup")
async def startup():
    ee.Initialize(project='savvy-ratio-490307-v3')
    logger.info("GEE ready")
    global carbon_model
    p = Path('models/carbon_model.pkl')
    if p.exists():
        with open(p, 'rb') as f:
            carbon_model = pickle.load(f)
        logger.info("ML model loaded")
    else:
        logger.warning("No model found — using rule-based fallback")

class AnalyzeRequest(BaseModel):
    coordinates: list

class CompareRequest(BaseModel):
    coordinates: list
    year1: int
    year2: int

class AvailableYearsRequest(BaseModel):
    coordinates: list

class ForecastRequest(BaseModel):
    coordinates:    list
    forecast_years: int = 3
    history_years:  int = 4

LULC_FACTORS = {
    10:120, 20:40, 30:15, 40:10,
    50:2,   60:1,  70:0,  80:0,
    90:80,  95:200, 100:5
}
LULC_LABELS = {
    10:'Tree cover', 20:'Shrubland',  30:'Grassland',
    40:'Cropland',   50:'Built-up',   60:'Bare/sparse',
    70:'Snow/ice',   80:'Water',      90:'Wetland',
    95:'Mangrove',  100:'Moss/lichen'
}

def get_recommendations(carbon, ndvi, lulc_class, change=None):
    recs = []
    if carbon < 30:
        recs.append("Low carbon stock detected. Consider reforestation.")
    if carbon > 80:
        recs.append("High carbon stock. Protect this area from deforestation.")
    if ndvi < 0.2:
        recs.append("Low NDVI. Vegetation is sparse. Plant green cover.")
    if ndvi > 0.6:
        recs.append("Healthy vegetation. Maintain current land use.")
    if lulc_class == 50:
        recs.append("Urban area. Carbon very low. Add urban green spaces.")
    if lulc_class == 10:
        recs.append("Forest area. High carbon sink. Protect from logging.")
    if lulc_class == 95:
        recs.append("Mangrove detected. Highest carbon density. Strictly protect.")
    if change is not None:
        if change < -10:
            recs.append(f"Carbon decreased by {abs(change):.1f} t/ha. Investigate deforestation.")
        if change > 10:
            recs.append(f"Carbon increased by {change:.1f} t/ha. Vegetation recovery observed.")
    if not recs:
        recs.append("Carbon levels are moderate. Monitor annually.")
    return recs

def rule_based(ndvi, lulc, temp, pop):
    base   = LULC_FACTORS.get(int(lulc), 10)
    ndvi_c = max(0, ndvi) * 30
    temp_p = max(0, min(40, temp - 20))
    pop_p  = min(50, pop * 5)
    return max(0.0, base + ndvi_c - temp_p - pop_p)

def extract_features(aoi, start_date, end_date):
    """
    Fetch all 4 features from GEE.
    FIXES:
    - cloud threshold relaxed to 30% for better coverage on older years
    - WorldPop always uses 2020 data (latest available in GEE)
    """
    def mask_s2(img):
        qa = img.select('QA60')
        m  = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        return img.updateMask(m).divide(10000)

    # Try 30% cloud threshold first
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .map(mask_s2))

    # Fallback to 50% if collection too small
    s2_col = ee.ImageCollection(ee.Algorithms.If(
        s2_col.size().gt(0),
        s2_col,
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
          .map(mask_s2)
    ))

    s2   = ee.Image(s2_col.median())
    ndvi = s2.normalizedDifference(['B8','B4']).rename('ndvi')
    rgb  = s2.select(['B4','B3','B2'])

    lulc = (ee.ImageCollection('ESA/WorldCover/v200')
            .first().select('Map').rename('lulc'))

    lst_col = (ee.ImageCollection('MODIS/061/MOD11A2')
               .filterBounds(aoi)
               .filterDate(start_date, end_date)
               .select('LST_Day_1km'))
    lst = ee.Image(ee.Algorithms.If(
        lst_col.size().gt(0),
        lst_col.mean().multiply(0.02).subtract(273.15),
        ee.Image.constant(25)
    )).rename('temperature')

    # WorldPop — ALWAYS 2020 (latest available in GEE)
    pop_col = (ee.ImageCollection('WorldPop/GP/100m/pop')
               .filterBounds(aoi)
               .filterDate('2020-01-01', '2020-12-31'))
    pop = ee.Image(ee.Algorithms.If(
        pop_col.size().gt(0),
        pop_col.mean().add(1).log(),
        ee.Image.constant(0)
    )).rename('population')

    stacked = ndvi.addBands(lulc).addBands(lst).addBands(pop).clip(aoi)
    stats   = stacked.reduceRegion(
        reducer    = ee.Reducer.mean(),
        geometry   = aoi,
        scale      = 500,
        maxPixels  = 1e9,
        bestEffort = True,
        tileScale  = 4
    ).getInfo()

    lulc_mode = lulc.reduceRegion(
        reducer    = ee.Reducer.mode(),
        geometry   = aoi,
        scale      = 100,
        maxPixels  = 1e9,
        bestEffort = True
    ).getInfo()

    return {
        'ndvi':        float(stats.get('ndvi')        or 0),
        'temperature': float(stats.get('temperature') or 25),
        'population':  float(stats.get('population')  or 0),
        'lulc_class':  int(lulc_mode.get('lulc')      or 30),
        'ndvi_img':    ndvi,
        'rgb_img':     rgb,
        'lulc_img':    lulc,
        'lst_img':     lst,
        'pop_img':     pop,
    }

def predict_carbon(ndvi, lulc_class, temperature, population):
    if carbon_model:
        row = pd.DataFrame([{
            'ndvi':        ndvi,
            'lulc':        lulc_class,
            'temperature': temperature,
            'population':  population
        }])
        return float(carbon_model.predict(row)[0]), "random_forest"
    return rule_based(ndvi, lulc_class, temperature, population), "rule_based"

def make_thumb(img, vis, aoi):
    return img.getThumbURL({
        **vis, 'region': aoi, 'dimensions': 512, 'format': 'png'
    })

def _year_date_range(year):
    return f"{year}-01-01", f"{year}-12-31"

def _validate_comparison_years(year1, year2):
    if year2 <= year1:
        raise ValueError(f"year2 must be after year1. Got {year1}, {year2}.")

def _comparison_statistics(v1, v2, threshold=2.0):
    diff = v2 - v1
    pct  = (diff / v1 * 100) if v1 > 0 else (100.0 if v2 > 0 else 0.0)
    if diff > threshold:
        status, trend = "gain", "increasing"
    elif diff < -threshold:
        status, trend = "loss", "decreasing"
    else:
        status, trend = "stable", "stable"
    return {
        "value_year1":         round(v1, 4),
        "value_year2":         round(v2, 4),
        "absolute_difference": round(diff, 4),
        "percentage_change":   round(pct, 2),
        "gain_loss_status":    status,
        "trend":               trend,
    }

def _process_comparison_year(aoi, year, area_ha):
    start_date, end_date = _year_date_range(year)
    feats = extract_features(aoi, start_date, end_date)
    carbon, model_used = predict_carbon(
        feats['ndvi'], feats['lulc_class'],
        feats['temperature'], feats['population'],
    )
    carbon = max(0.0, min(500.0, carbon))
    carbon_img = make_carbon_image(
        feats['lulc_img'], feats['ndvi_img'],
        feats['lst_img'],  feats['pop_img'], aoi,
    )
    return {
        "year":         year,
        "start_date":   start_date,
        "end_date":     end_date,
        "feats":        feats,
        "carbon":       carbon,
        "model_used":   model_used,
        "carbon_img":   carbon_img,
        "carbon_total": round(carbon * area_ha, 1),
    }

def make_carbon_image(lulc_img, ndvi_img, lst_img, pop_img, aoi):
    cf = (lulc_img
          .where(lulc_img.eq(10),120).where(lulc_img.eq(20), 40)
          .where(lulc_img.eq(30), 15).where(lulc_img.eq(40), 10)
          .where(lulc_img.eq(50),  2).where(lulc_img.eq(60),  1)
          .where(lulc_img.eq(70),  0).where(lulc_img.eq(80),  0)
          .where(lulc_img.eq(90), 80).where(lulc_img.eq(95),200)
          .where(lulc_img.eq(100), 5))
    return (cf
            .add(ndvi_img.clamp(0,1).multiply(30))
            .subtract(lst_img.subtract(20).clamp(0,40))
            .subtract(pop_img.multiply(5).clamp(0,50))
            .max(0).clip(aoi))

def make_classified_change_image(carbon_img1, carbon_img2, aoi):
    """
    Classified change detection — thresholds based on meaningful carbon change.
    gain  > +5 t/ha  → green
    loss  < -5 t/ha  → red
    stable -5 to +5  → transparent (value 0)
    """
    diff = carbon_img2.subtract(carbon_img1)
    # Classify into 3 zones
    classified = (ee.Image(0)               # stable = 0
                  .where(diff.gt(5),  1)    # gain   = 1
                  .where(diff.lt(-5),-1))   # loss   = -1
    return classified.clip(aoi)

def get_available_years_for_aoi(aoi):
    candidate_years = [2017,2018,2019,2020,2021,2022,2023,2024,2025]
    def check_year(year):
        try:
            start, end = f"{year}-01-01", f"{year}-12-31"
            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(aoi).filterDate(start, end)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
            count = col.limit(1).size().getInfo()
            return year, count > 0
        except Exception as e:
            logger.error(f"Error checking year {year}: {e}")
            return year, False
    with ThreadPoolExecutor(max_workers=len(candidate_years)) as executor:
        results = list(executor.map(check_year, candidate_years))
    return sorted([y for y, ok in results if ok])

# ================================================================
# ENDPOINT — /available-years
# ================================================================
@app.post("/available-years")
async def available_years(req: AvailableYearsRequest):
    try:
        aoi   = ee.Geometry.Polygon([req.coordinates])
        years = get_available_years_for_aoi(aoi)
        return {"available_years": years}
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================
# ENDPOINT 1 — /analyze
# ================================================================
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        year       = datetime.now().year - 1
        aoi        = ee.Geometry.Polygon([req.coordinates])
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        area_ha    = aoi.area(maxError=1).divide(10000).getInfo()

        feats = extract_features(aoi, start_date, end_date)
        carbon, model_used = predict_carbon(
            feats['ndvi'], feats['lulc_class'],
            feats['temperature'], feats['population']
        )
        carbon       = max(0.0, min(500.0, carbon))
        carbon_level = "High" if carbon > 80 else "Medium" if carbon > 30 else "Low"
        recs         = get_recommendations(carbon, feats['ndvi'], feats['lulc_class'])
        carbon_img   = make_carbon_image(
            feats['lulc_img'], feats['ndvi_img'],
            feats['lst_img'],  feats['pop_img'], aoi
        )

        return {
            "year":                year,
            "carbon":              round(carbon, 2),
            "carbon_total_tonnes": round(carbon * area_ha, 1),
            "carbon_level":        carbon_level,
            "ndvi_mean":           round(feats['ndvi'], 4),
            "temperature_mean":    round(feats['temperature'], 2),
            "population_mean":     round(feats['population'], 4),
            "lulc_class":          feats['lulc_class'],
            "lulc_label":          LULC_LABELS.get(feats['lulc_class'], "Unknown"),
            "area_ha":             round(area_ha, 2),
            "model_used":          model_used,
            "recommendations":     recs,
            "rgb":        make_thumb(feats['rgb_img'].clip(aoi),
                            {'min':0,'max':0.3,'gamma':1.4,'bands':['B4','B3','B2']}, aoi),
            "ndvi":       make_thumb(feats['ndvi_img'].clip(aoi),
                            {'min':-0.2,'max':0.9,'palette':['brown','white','darkgreen']}, aoi),
            "lulc":       make_thumb(feats['lulc_img'].clip(aoi),
                            {'min':10,'max':100,
                             'palette':['006400','ffbb22','ffff4c','f096ff','fa0000',
                                        'b4b4b4','f0f0f0','0064c8','0096a0','00cf75','fae6a0']}, aoi),
            "carbon_map": make_thumb(carbon_img,
                            {'min':0,'max':150,'palette':['red','yellow','green']}, aoi),
            "stats": {
                "year":         year,
                "carbon_t_ha":  round(carbon, 2),
                "carbon_total": round(carbon * area_ha, 1),
                "ndvi_mean":    round(feats['ndvi'], 4),
                "temp_c":       round(feats['temperature'], 2),
                "area_ha":      round(area_ha, 2),
                "lulc_label":   LULC_LABELS.get(feats['lulc_class'], "Unknown"),
                "model_used":   model_used,
                "carbon_level": carbon_level,
            }
        }
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================
# ENDPOINT 2 — /compare  (fixed comparison logic)
# ================================================================
@app.post("/compare")
async def compare(req: CompareRequest):
    try:
        _validate_comparison_years(req.year1, req.year2)
        aoi     = ee.Geometry.Polygon([req.coordinates])
        area_ha = aoi.area(maxError=1).divide(10000).getInfo()

        y1 = _process_comparison_year(aoi, req.year1, area_ha)
        y2 = _process_comparison_year(aoi, req.year2, area_ha)

        c1 = y1['carbon'];  c2 = y2['carbon']
        f1 = y1['feats'];   f2 = y2['feats']

        carbon_stats = _comparison_statistics(c1, c2, threshold=0.5)
        ndvi_stats   = _comparison_statistics(f1['ndvi'], f2['ndvi'], threshold=0.01)

        change     = carbon_stats['absolute_difference']
        change_pct = carbon_stats['percentage_change']
        trend      = carbon_stats['trend']
        recs       = get_recommendations(c2, f2['ndvi'], f2['lulc_class'], change)

        # Classified change map: gain/loss/stable with meaningful thresholds
        classified_img = make_classified_change_image(
            y1['carbon_img'], y2['carbon_img'], aoi
        )

        # Visualize classified map:
        # -1 = loss = red, 0 = stable = transparent, 1 = gain = green
        classified_thumb = classified_img.visualize(**{
            'min': -1, 'max': 1,
            'palette': ['#d62728', '#aec7e8', '#2ca02c']
        }).updateMask(classified_img.neq(0)).getThumbURL({
            'region': aoi, 'dimensions': 512, 'format': 'png'
        })

        carbon_map_y1 = make_thumb(y1['carbon_img'],
            {'min':0,'max':150,'palette':['red','yellow','green']}, aoi)
        carbon_map_y2 = make_thumb(y2['carbon_img'],
            {'min':0,'max':150,'palette':['red','yellow','green']}, aoi)
        rgb_y1 = make_thumb(f1['rgb_img'].clip(aoi),
            {'min':0,'max':0.3,'gamma':1.4,'bands':['B4','B3','B2']}, aoi)
        rgb_y2 = make_thumb(f2['rgb_img'].clip(aoi),
            {'min':0,'max':0.3,'gamma':1.4,'bands':['B4','B3','B2']}, aoi)

        return {
            # Core values
            "year1":            req.year1,
            "year2":            req.year2,
            "carbon_year1":     round(c1, 2),
            "carbon_year2":     round(c2, 2),
            "change":           change,
            "change_pct":       change_pct,
            "trend":            trend,
            "ndvi_year1":       round(f1['ndvi'], 4),
            "ndvi_year2":       round(f2['ndvi'], 4),
            "ndvi_change":      ndvi_stats['absolute_difference'],
            "lulc_year1":       LULC_LABELS.get(f1['lulc_class'], 'Unknown'),
            "lulc_year2":       LULC_LABELS.get(f2['lulc_class'], 'Unknown'),
            "temp_year1":       round(f1['temperature'], 2),
            "temp_year2":       round(f2['temperature'], 2),
            "area_ha":          round(area_ha, 2),
            "model_used":       y2['model_used'],
            "recommendations":  recs,
            # Map thumbnails
            "classified_change_map": classified_thumb,  # main default map
            "carbon_map_year1":      carbon_map_y1,
            "carbon_map_year2":      carbon_map_y2,
            "rgb_year1":             rgb_y1,
            "rgb_year2":             rgb_y2,
            # Legacy fields for frontend compatibility
            "carbon_year1_map":      carbon_map_y1,
            "carbon_year2_map":      carbon_map_y2,
            "change_map":            classified_thumb,
            "rgb_year2":             rgb_y2,
            # Stats block
            "stats": {
                "year1": {
                    "carbon": round(c1,2), "ndvi": round(f1['ndvi'],4),
                    "temp":   round(f1['temperature'],2),
                    "lulc":   LULC_LABELS.get(f1['lulc_class'],'Unknown'),
                },
                "year2": {
                    "carbon": round(c2,2), "ndvi": round(f2['ndvi'],4),
                    "temp":   round(f2['temperature'],2),
                    "lulc":   LULC_LABELS.get(f2['lulc_class'],'Unknown'),
                },
                "change":     change,
                "change_pct": change_pct,
                "trend":      trend,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================
# ENDPOINT 3 — /forecast  (real GEE per polygon)
# ================================================================
@app.post("/forecast")
async def forecast(req: ForecastRequest):
    try:
        aoi          = ee.Geometry.Polygon([req.coordinates])
        current_year = datetime.now().year - 1
        years_hist   = list(range(
            current_year - req.history_years + 1,
            current_year + 1
        ))
        carbons_hist = []
        for yr in years_hist:
            feats = extract_features(aoi, f"{yr}-01-01", f"{yr}-12-31")
            carbon, _ = predict_carbon(
                feats['ndvi'], feats['lulc_class'],
                feats['temperature'], feats['population']
            )
            carbons_hist.append(round(max(0.0, min(500.0, carbon)), 2))
            logger.info(f"Forecast {yr} → {carbons_hist[-1]} t/ha")

        X   = np.array(years_hist).reshape(-1, 1)
        y   = np.array(carbons_hist)
        reg = LinearRegression()
        reg.fit(X, y)

        slope     = float(reg.coef_[0])
        intercept = float(reg.intercept_)
        r2        = float(reg.score(X, y))

        last_year    = max(years_hist)
        future_years = list(range(last_year+1, last_year+1+req.forecast_years))
        future_preds = reg.predict(np.array(future_years).reshape(-1,1))
        forecast_vals = [round(max(0.0, float(v)), 2) for v in future_preds]

        trend_dir = ("increasing" if slope > 0.5 else
                     "decreasing" if slope < -0.5 else "stable")

        return {
            "historical": {"years": years_hist,  "carbon": carbons_hist},
            "forecast":   {"years": future_years, "carbon": forecast_vals},
            "model":      {"slope": round(slope,3), "intercept": round(intercept,2), "r2": round(r2,4)},
            "trend":       trend_dir,
            "description": f"Carbon is {trend_dir} by {abs(slope):.2f} t/ha per year",
            "source":      "GEE real satellite data + linear regression",
        }
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================
# HEALTH CHECK
# ================================================================
@app.get("/health")
async def health():
    return {"status":"ok","model":"random_forest" if carbon_model else "rule_based"}