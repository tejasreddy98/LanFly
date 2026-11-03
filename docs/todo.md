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
- [ ] **9. Assign Host-Plants to  counties**--> map county to known vegetations and do another effort analysis to look at proportion of vegetations and how they change with lanternfly observations. Help to determine if this is a feature we want to include in our model

**Order:** 1–3 decide the response variable (first model due 10/30); 4–6 inform predictors; 7–8 shape cleaning and validation.


## Modeling ToDos (`notebooks/03_regularized_regression.ipynb`)

Preliminary Ridge / Lasso / Elastic Net on `log1p(slf_obs)` with log effort as a predictor; invasion region, county-years with ≥200 observations; train 2015–2022, test 2023–2025.

- [ ] Team interpretation written for each section (cells marked `_TODO_`)
- [ ] Decide whether feature set B's spread feature (last year's count) belongs in the final model (README Open Decision 6)
- [ ] Feature set A chose no penalty (alpha at the bottom of the grid): discuss whether regularization is needed for these predictors, or only once more features are added
- [ ] Try a regularized Poisson / negative binomial model on `slf_obs` with log effort as a comparison; also a two-part model for the zeros (88% of county-years)
- [ ] Revisit the EDA 1b note: the log-log slope is about 0.51, so an effort *offset* (which fixes the slope at 1) isn't supported. Bears on the response-variable decision in EDA section 1.
- [ ] Add a spatial holdout (e.g., hold out states) alongside the temporal one
