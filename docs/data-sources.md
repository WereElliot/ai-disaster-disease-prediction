# dataSources.md
# Climate and Health Data Sources Documentation
## AI Disease and Disaster Outbreak Prediction Platform

**Last Updated**: March 2026
**Maintained by**: [Code Servants]
**Data Coordinator**: [Contact email]

---------------------------------------

## Table of Contents
1. [Climate Data Sources](#1-climate-data-sources)
2. [Disease Incidence Data](#2-disease-incidence-data)
3. [Disaster Records](#3-disaster-records)
4. [Demographic and Contextual Data](#4-demographic-and-contextual-data)
5. [Data Quality Assessment](#5-data-quality-assessment)
6. [Access Protocols](#6-access-protocols)

----------------------------------------

## 1. Climate Data Sources

### 1.1 NOAA (National Oceanic and Atmospheric Administration)
**Description**  Global historical weather data, climate forecasts, and extreme weather events
**Data types** Temperature, precipitation, wind speed, humidity, atmospheric pressure
**Temporal Coverage** 2017-present
**Spatial Coverage**  Global, various resolutions
**Access Method** API, FTP, direct download
**API Endpoint**  `https://www.ncdc.noaa.gov/cdo-web/webservices/v2`
**Authentication** Free API token required
**Update Frequency** Daily to monthly depending on product
**License** Public domain (US Government work)
**Citation** NOAA National Centers for Environmental Information
**Key Products** GHCN-Daily, GSOD, Climate Normals

### 1.2 NASA Earth Data (POWER Project)
**Description** Satellite-derived meteorological and solar radiation data
**Data Types** Solar radiation, temperature, humidity, wind, precipitation
**Temporal Coverage** 1981-present
**Spatial Coverage** Global, 0.5° x 0.5° resolution
**Access Method** REST API, web interface
**API Endpoint** `https://power.larc.nasa.gov/api/power/`
**Authentication** None required
**Update Frequency** Daily (near real-time)
**License** Open data policy, NASA Earth Science Data
**Citation** NASA Langley Research Center POWER Project

### 1.3 ERA5 (ECMWF Reanalysis)
**Description** Fifth generation ECMWF atmospheric reanalysis of the global climate
**Data Types** Hourly estimates of atmospheric, land, and oceanic variables
**Temporal Coverage** 1950-present
**Spatial Coverage** Global, 0.25° x 0.25° resolution
**Access Method** Climate Data Store (CDS) API, Copernicus
**API Endpoint** `https://cds.climate.copernicus.eu/api/v2`
**Authentication** Free registration required
**Update Frequency** Monthly with 5-day latency
**License** Copernicus License (Free, with attribution)
**Citation** Hersbach et al. (2023), ERA5 monthly averaged data

### 1.4 CHIRPS (Climate Hazards Group InfraRed Precipitation with Station Data)
**Description** High-resolution precipitation climatology
**Data Types** Rainfall estimates
**Temporal Coverage** 1981-present
**Spatial Coverage** Global (50°S-50°N), 0.05° resolution
**Access Method** FTP, Google Earth Engine
**API Endpoint** `https://data.chc.ucsb.edu/products/CHIRPS-2.0/`
**Authentication** None
**Update Frequency** Monthly (preliminary), Final after 3 weeks
**License** Free for research and humanitarian use
**Citation** Funk et al. (2015), Scientific Data

## 2. Disease Incidence Data

### 2.1 World Health Organization (WHO)
**Description** Global health observatory, disease surveillance data
**Data Types** Malaria, cholera, dengue, measles, meningitis, etc.
**Temporal Coverage** 2000-present (varies by disease)
**Spatial Coverage** Country-level (some subnational)
**Access Method** Global Health Observatory API, data download
**API Endpoint** `https://www.who.int/data/gho/info/gho-odata-api`
**Authentication** None required
**Update Frequency** Quarterly to annually
**License** CC BY-NC-SA 3.0 IGO
**Limitations** Country-level only, reporting delays
**Specific Datasets**:
- Malaria: World Malaria Report
- Cholera: Weekly Epidemiological Record
- Dengue: Dengue surveillance data

### 2.2 HealthMap
**Description** Real-time infectious disease surveillance system
**Data Types** Outbreak reports, news alerts, official reports
**Diseases Covered** All infectious diseases including malaria, cholera, dengue
**Temporal Coverage** 2006-present (real-time)
**Spatial Coverage** Global, city-level when available
**Access Method** API, RSS feeds
**API Endpoint** `https://www.healthmap.org/site/api`
**Authentication** Free API key required for high-volume access
**Update Frequency** Real-time (multiple times daily)
**License** Free for research with attribution
**Citation** HealthMap, Boston Children's Hospital

### 2.3 EIDR (Epidemic Intelligence from Digital Resources)
**Description** Epidemic data curated from multiple sources
**Data Types** Outbreak reports, case counts, geographic spread
**Diseases Covered** Comprehensive including emerging diseases
**Temporal Coverage** 2015-present
**Spatial Coverage**  Global, subnational
**Access Method** Direct download
**Website** `https://eidr.org`
**Authentication** Registration required
**Update Frequency** Weekly
**License** Research use only
**Note** Partnership with WHO and ministries

### 2.4 Local Ministries of Health
**Kenya** MoH Open Data Portal
**Data Tyoes** Malaria, cholera, surveillance
**Update frequency** Monthly
**Access Requirements** Registration

### 2.5 ProMED-mail
**Description** Program for Monitoring Emerging Diseases
**Data Types** Unstructured outbreak reports, expert commentary
**Temporal Coverage** 1994-present
**Spatial Coverage** Global
**Access Method** Email listserv, website, API
**Website** `https://promedmail.org`
**Authentication** Free subscription
**Update Frequency** Multiple times daily
**License** Free for research with attribution
**Use Case** Early warning, qualitative context

## 3. Disaster Records

### 3.1 Flood Data

#### Global Flood Database
**Description** Satellite-derived flood extent and population exposure
**Data Types** Flood maps, inundation duration
**Temporal Coverage** 2000-2018 (expanding)
**Spatial Coverage** Global
**Access Method** Download, Google Earth Engine
**Source** `https://global-flood-database.cloudtostreet.ai/`
**Authentication** None
**Update Frequency** Periodic
**License** CC BY 4.0
**Citation** Tellman et al. (2021), Nature

#### Dartmouth Flood Observatory
**Description** Global active flood detection
**Data Types** Flood event archive, severity, affected area
**Temporal Coverage** 1985-present
**Spatial Coverage** Global
**Access Method** Web portal, shapefiles
**Website** `https://floodobservatory.colorado.edu/`
**Authentication** None
**Update Frequency** Real-time during events
**License** Free for research

### 3.2 Drought Data

#### U.S. Drought Monitor / Global Drought Information System
**Description** Drought classification and monitoring
**Data Types** Drought intensity (D0-D4), duration, extent
**Temporal Coverage** 2000-present
**Spatial Coverage** Global (various products)
**Access Method** Web services, download 
**Website** `https://droughtmonitor.unl.edu/`
**Authentication** None
**Update Frequency** Weekly
**License** Public domain

#### SPEI Global Drought Monitor
**Description** Standardized Precipitation-Evapotranspiration Index
**Data Types** SPEI timeseries at multiple scales
**Temporal Coverage** 1901-2022
**Spatial Coverage** Global, 0.5° resolution
**Access Method** Download, THREDDS server
**Source** `https://spei.csic.es/`
**Authentication** None
**Update Frequency** Monthly
**License** Free for research

## 4. Demographic and Contextual Data

### 4.1 WorldPop
**Description** High-resolution population distribution
**Data Types** Population counts, age/sex structures, urban/rural
**Temporal Coverage** 2000-2020
**Spatial Coverage** Global, 100m resolution
**Access Method** Download, API
**Source** `https://www.worldpop.org/`
**Authentication** None
**Update Frequency** Annual
**License** CC BY 4.0

### 4.2 Gridded Population of the World (GPW)
**Description** NASA SEDAC population grids
**Data Types** Population density, counts
**Temporal Coverage** 2000, 2005, 2010, 2015, 2020
**Spatial Coverage** Global, 1km resolution
**Access Method** Download
**Source** `https://sedac.ciesin.columbia.edu/`
**Authentication** Registration required
**Update Frequency** Every 5 years
**License** Free for non-commercial

## 5. Data Quality Assessment

### Data Quality Checks
- Completeness: % missing values by region/time
- Temporal consistency: Break detection
- Spatial coverage: Representativeness
- Cross-validation: Multiple source comparison

## 6. Access Protocols

### 6.1 Automated Access Scripts

#### Python Template for ERA5 Access
```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': ['2m_temperature', 'total_precipitation'],
        'year': ['2020', '2021', '2022', '2023'],
        'month': ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'],
        'time': '00:00',
        'format': 'netcdf',
    },
    'era5_download.nc')