import ee
from ee_plugin import Map

# zoom in somewhere
bounds = Map.getBounds(True)


# Load S2 data
def maskS2clouds(image):
    qa = image.select('QA60');
    cloudBitMask = 1 << 10;
    cirrusBitMask = 1 << 11;
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0));
    return image.updateMask(image.mask()).divide(10000)

imageS2_01jun2019 =  ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
    .filterDate('2019-06-01', '2019-06-30')\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50))\
    .map(maskS2clouds);


# Visualize 
Map.addLayer(imageS2_01jun2019.median(), {'min': 0.0, 'max': 0.12,  'bands': ['B4', 'B3', 'B2']}, 'Sentinel-2 RGB 01jun2019 median', True, 1.0);
