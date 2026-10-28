Team ToDos

1) review the code and confirm the iNaturalist query filters (research grade, US) are the ones we want.
2) Review and handle missing data: (example: NOAA’s files mark missing temperatures as -99.90, not the -99.99 its documentation says.)
3) 


## EDA ToDos (`notebooks/02_eda.ipynb`)

Each item: question --> chart(s) --> decision/problem it helps solve. (We'll need to write interpretations for each cell).

- [x] **1. Observer effort**: do sightings track lanternflies or iNaturalist usage? --> effort trends, log-log scatter, count vs. rate maps --> response variable choice
  - [ ] Team interpretation written
  - [ ] Check Arlington, VA outlier (13% of all observations are SLF; possible targeted campaign)
- [ ] **2. Invasion spread**: how has SLF spread since 2014? --> map of first-detection year by county; new counties per year --> scope (region/years); possible "will county X be invaded next year?" response
- [ ] **3. Response distribution**: share of zeros, skew, mean vs. variance --> Poisson vs. negative binomial, transforms, two-part model
- [ ] **4. Climate vs. invasion**: do invaded counties differ in climate from not-yet-invaded counties near the front? --> distributions of winter min temp, avg temp, precipitation --> hypotheses and predictor list
- [ ] **5. Predictor correlation**: how correlated are the climate variables? --> correlation heatmap --> PCA / regularization justification
- [ ] **6. Seasonality**: when in the year are SLF seen, by latitude? --> sightings by month/year --> which climate months matter; year vs. season time step
- [ ] **7. Data quality sensitivity**: do results change when dropping obscured or low-accuracy coordinates? --> county counts under different cutoffs --> cleaning thresholds
- [ ] **8. Spatial clustering**: are neighboring counties similar? --> county vs. neighbor-average scatter --> validation design (spatial/temporal holdout)

**Order:** 1–3 decide the response variable (first model due 10/30); 4–6 inform predictors; 7–8 shape cleaning and validation.
