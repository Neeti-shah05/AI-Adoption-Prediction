# ALY 6040 — Final Project 

# ===== 0) Installing the Packages =====
suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(janitor)
  library(skimr)
  library(corrplot)
  library(recipes)
  library(ranger)
  library(pROC)
  library(caret)
  library(PRROC)
  library(glmnet)
  library(Matrix)
  library(xgboost)
  library(ggplot2)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b
set.seed(123)
options(stringsAsFactors = FALSE)

# ===== 1) Inputs / Toggles =====
FAST_MODE          <- TRUE        # TRUE = quick search; FALSE = wider search
USE_CLASS_WEIGHTS  <- TRUE        # Used for RF (ranger)
SAVE_OUTPUTS       <- TRUE        # save figures/tables to ./outputs

DATA_PATH   <- "C:/Users/neeti/OneDrive/Desktop/NEU/ALY 6040/ai_adoption_dataset.xlsx"
SHEET       <- 1
TARGET_COL  <- NULL               # if NULL, last column is target
ID_COLS     <- NULL               # will auto-detect below after clean_names()

if (SAVE_OUTPUTS) dir.create("outputs", showWarnings = FALSE)

# ===== 2) Load & Clean =====
df_raw <- read_excel(DATA_PATH, sheet = SHEET) %>% as.data.frame()
df     <- df_raw %>% clean_names()

# Guess ID columns if not supplied
if (is.null(ID_COLS)) {
  guess_ids <- c("company_id","companyid","id")
  ID_COLS <- intersect(names(df), guess_ids)
}

# Target guess
if (is.null(TARGET_COL)) TARGET_COL <- names(df)[ncol(df)]
stopifnot(TARGET_COL %in% names(df))

df <- df %>%
  mutate(across(where(is.character), ~ trimws(.x))) %>%
  mutate(across(where(is.character), ~ na_if(.x, ""))) %>%
  mutate(across(where(is.character), as.factor)) %>%
  relocate(all_of(TARGET_COL), .after = last_col())

# ===== 3) EDA =====
cat("\n===== BASIC STRUCTURE =====\n"); print(glimpse(df))
cat("\n===== QUICK SUMMARY (skim) =====\n"); print(skimr::skim(df))
cat("\n===== MISSING BY COLUMN =====\n"); print(sapply(df, function(x) sum(is.na(x))))

num_vars <- names(dplyr::select(df, where(is.numeric)))
if (length(num_vars) >= 2) {
  cmat <- df %>% dplyr::select(all_of(num_vars)) %>%
    mutate(across(everything(), as.numeric)) %>%
    cor(use = "pairwise.complete.obs")
  corrplot(cmat, method = "color", type = "upper", addCoef.col = "black",
           tl.col = "black", tl.srt = 45, number.cex = 0.5,
           mar = c(0,0,1,0), title = "Correlation Heatmap (Numeric Features)")
  if (SAVE_OUTPUTS) {
    grDevices::dev.copy(png, filename = "outputs/01_corr_heatmap.png", width = 1600, height = 1200, res = 180)
    dev.off()
  }
}

# ===== 4) Classification prep =====
y <- df[[TARGET_COL]]
stopifnot(!is.numeric(y))        # expecting classification
if (!is.factor(df[[TARGET_COL]])) df[[TARGET_COL]] <- as.factor(df[[TARGET_COL]])

# If binary, make rarer class positive (first)
if (nlevels(df[[TARGET_COL]]) == 2) {
  tab <- table(df[[TARGET_COL]]); pos <- names(sort(tab))[1]
  df[[TARGET_COL]] <- relevel(df[[TARGET_COL]], ref = pos)
  cat("Positive class set to:", pos, "\n")
}

# ===== 5) Split: Train (60%) / Validation (20%) / Test (20%) =====
set.seed(123)
idx_train <- caret::createDataPartition(df[[TARGET_COL]], p = 0.6, list = FALSE)
train_df  <- df[idx_train, ]
hold_df   <- df[-idx_train, ]

set.seed(123)
idx_valid <- caret::createDataPartition(hold_df[[TARGET_COL]], p = 0.5, list = FALSE)
valid_df  <- hold_df[idx_valid, ]
test_df   <- hold_df[-idx_valid, ]

# ===== 6) Shared Recipe =====
rec <- recipe(as.formula(paste(TARGET_COL, "~ .")), data = train_df) %>%
  update_role(all_of(ID_COLS), new_role = "ID") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01, other = "other") %>% # rare levels
  step_nzv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

set.seed(123)
prep_rec <- prep(rec, training = train_df, retain = TRUE, verbose = FALSE)

baked_train <- bake(prep_rec, new_data = train_df) %>% select(-any_of(ID_COLS))
baked_valid <- bake(prep_rec, new_data = valid_df) %>% select(-any_of(ID_COLS))
baked_test  <- bake(prep_rec, new_data = test_df)  %>% select(-any_of(ID_COLS))

# Targets & predictors
y_train <- droplevels(baked_train[[TARGET_COL]])
X_train <- baked_train %>% select(-all_of(TARGET_COL)) %>% mutate(across(everything(), as.numeric))
y_valid <- factor(baked_valid[[TARGET_COL]], levels = levels(y_train))
X_valid <- baked_valid %>% select(-all_of(TARGET_COL)) %>% mutate(across(everything(), as.numeric))
y_test  <- factor(baked_test[[TARGET_COL]],  levels = levels(y_train))
X_test  <- baked_test  %>% select(-all_of(TARGET_COL)) %>% mutate(across(everything(), as.numeric))

# Ensure aligned columns
X_valid <- X_valid %>% select(all_of(names(X_train)))
X_test  <- X_test  %>% select(all_of(names(X_train)))

# ===== 7) Shared Metric Helpers =====
macro_auc_valid <- function(probs, y_true) {
  a <- c()
  for (cls in levels(y_true)) {
    biny <- factor(y_true == cls, levels = c(FALSE, TRUE))
    roc_obj <- try(pROC::roc(biny, probs[, cls], quiet = TRUE), silent = TRUE)
    if (!inherits(roc_obj, "try-error")) a[cls] <- as.numeric(pROC::auc(roc_obj))
  }
  mean(a, na.rm = TRUE)
}
ovr_pr_auc <- function(prob_mat, y_true) {
  out <- c()
  for (cls in levels(y_true)) {
    scores_pos <- prob_mat[, cls][y_true == cls]
    scores_neg <- prob_mat[, cls][y_true != cls]
    pr_obj <- pr.curve(scores.class0 = scores_pos, scores.class1 = scores_neg, curve = FALSE)
    out[cls] <- as.numeric(pr_obj$auc.integral)
  }
  out
}
multiclass_logloss <- function(prob_mat, y_true, eps = 1e-15) {
  probs <- pmax(pmin(prob_mat, 1 - eps), eps)
  idx <- cbind(seq_along(y_true), match(y_true, colnames(probs)))
  -mean(log(probs[idx]))
}
macro_micro_f1 <- function(cm_table) {
  classes <- colnames(cm_table)
  tp <- diag(cm_table)
  fp <- rowSums(cm_table) - tp
  fn <- colSums(cm_table) - tp
  prec <- tp / pmax(tp + fp, 1e-9)
  rec  <- tp / pmax(tp + fn, 1e-9)
  f1   <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  macro_f1 <- mean(f1, na.rm = TRUE)
  TP <- sum(tp); FP <- sum(fp); FN <- sum(fn)
  prec_micro <- TP / pmax(TP + FP, 1e-9)
  rec_micro  <- TP / pmax(TP + FN, 1e-9)
  micro_f1 <- ifelse(prec_micro + rec_micro > 0, 2 * prec_micro * rec_micro / (prec_micro + rec_micro), 0)
  list(per_class_f1 = setNames(f1, classes), macro_f1 = macro_f1, micro_f1 = micro_f1)
}
tune_thresholds_ovr <- function(prob_mat, y_true, grid = seq(0.05, 0.95, by = 0.05)) {
  best_t <- rep(0.5, ncol(prob_mat)); names(best_t) <- colnames(prob_mat)
  for (j in seq_len(ncol(prob_mat))) {
    cls <- colnames(prob_mat)[j]
    y_bin <- factor(y_true == cls, levels = c(FALSE, TRUE))
    pj <- prob_mat[, j]
    best_f1 <- -Inf; best_th <- 0.5
    for (t in grid) {
      pred <- factor(ifelse(pj >= t, TRUE, FALSE), levels = c(FALSE, TRUE))
      tp <- sum(pred == TRUE & y_bin == TRUE)
      fp <- sum(pred == TRUE & y_bin == FALSE)
      fn <- sum(pred == FALSE & y_bin == TRUE)
      prec <- tp / pmax(tp + fp, 1e-9)
      rec  <- tp / pmax(tp + fn, 1e-9)
      f1   <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
      if (f1 > best_f1) { best_f1 <- f1; best_th <- t }
    }
    best_t[j] <- best_th
  }
  best_t
}
predict_with_thresholds <- function(prob_mat, thresholds) {
  adj <- sweep(prob_mat, 2, thresholds, "/")
  classes <- colnames(adj)
  factor(classes[max.col(adj, ties.method = "first")], levels = classes)
}
calibration_df <- function(scores, y_bin, bins = 10) {
  tibble(score = scores, y = y_bin) %>%
    mutate(bin = ntile(score, bins)) %>%
    group_by(bin) %>%
    summarise(mean_pred = mean(score), emp_rate = mean(y), n = dplyr::n(), .groups = "drop") %>%
    arrange(mean_pred)
}

# ===== 8) Parallel / RF grid setup (unchanged) =====
p <- ncol(X_train)
if (FAST_MODE) {
  grid <- expand.grid(
    num.trees       = 200,
    mtry            = unique(pmax(1, round(c(sqrt(p), p/4, p/2)))),
    min.node.size   = c(5, 10),
    sample.fraction = c(0.6, 0.7)
  )
  TOP_K <- 4
  FINAL_TREES <- 800
} else {
  grid <- expand.grid(
    num.trees       = c(300, 500),
    mtry            = unique(pmax(1, round(c(sqrt(p), p/3, p/5, p/2)))),
    min.node.size   = c(3, 7, 12),
    sample.fraction = c(0.6, 0.7, 0.8)
  )
  TOP_K <- 6
  FINAL_TREES <- 1200
}
nthreads <- max(1, parallel::detectCores() - 1)

# Prepare metrics container
metrics_list <- list()

# =====================================================================
# ==========================  RANDOM FOREST  ===========================
# =====================================================================
# Class weights
cw <- NULL
if (USE_CLASS_WEIGHTS) {
  tbl <- table(y_train)
  w <- as.numeric(1 / tbl)
  w <- w / mean(w)
  names(w) <- names(tbl)
  cw <- w
  cat("\nClass weights used (RF):\n"); print(round(cw, 3))
}

# Stage A — OOB shortlist
stageA <- tibble()
for (i in seq_len(nrow(grid))) {
  g <- grid[i,]
  oob_err <- Inf
  try({
    fit <- ranger(
      formula = .outcome ~ .,
      data = data.frame(.outcome = y_train, X_train),
      num.trees = g$num.trees,
      mtry = g$mtry,
      min.node.size = g$min.node.size,
      sample.fraction = g$sample.fraction,
      splitrule = "gini",
      probability = TRUE,
      oob.error = TRUE,
      importance = "none",
      num.threads = nthreads,
      class.weights = cw,
      save.memory = TRUE,
      seed = 123
    )
    oob_err <- fit$prediction.error
  }, silent = TRUE)
  stageA <- bind_rows(stageA, tibble(
    i = i,
    num.trees = g$num.trees, mtry = g$mtry, min.node.size = g$min.node.size,
    sample.fraction = g$sample.fraction,
    oob_err = oob_err
  ))
}
stageA <- arrange(stageA, oob_err)
cat("\n[RF] STAGE A (OOB shortlist):\n"); print(stageA)

# Stage B — validate top configs
cands <- head(stageA, TOP_K)
stageB <- tibble(); best <- NULL
for (k in seq_len(nrow(cands))) {
  g <- cands[k,]
  fit <- ranger(
    formula = .outcome ~ .,
    data = data.frame(.outcome = y_train, X_train),
    num.trees = g$num.trees,
    mtry = g$mtry,
    min.node.size = g$min.node.size,
    sample.fraction = g$sample.fraction,
    splitrule = "gini",
    probability = TRUE,
    num.threads = nthreads,
    class.weights = cw,
    save.memory = TRUE,
    seed = 123
  )
  pred_valid <- predict(fit, data = X_valid)$predictions
  colnames(pred_valid) <- levels(y_train)
  aucv <- if (nlevels(y_valid) == 2) {
    pos_class <- levels(y_valid)[1]
    as.numeric(pROC::auc(pROC::roc(y_valid == pos_class, pred_valid[, pos_class], quiet = TRUE)))
  } else {
    macro_auc_valid(pred_valid, y_valid)
  }
  row <- tibble(
    num.trees = g$num.trees, mtry = g$mtry, min.node.size = g$min.node.size,
    sample.fraction = g$sample.fraction, val_auc = aucv
  )
  stageB <- bind_rows(stageB, row)
  if (is.null(best) || aucv > best$val_auc) best <- as.list(row)
}
cat("\n[RF] VALIDATION results (Stage B):\n"); print(arrange(stageB, desc(val_auc)))
cat("\n[RF] Selected (by VALID AUC):\n"); print(best)

# Final RF fit (Train+Valid)
X_tv <- rbind(X_train, X_valid)
y_tv <- factor(c(as.character(y_train), as.character(y_valid)), levels = levels(y_train))

rf_final <- ranger(
  formula = .outcome ~ .,
  data = data.frame(.outcome = y_tv, X_tv),
  num.trees = FINAL_TREES,
  mtry = best$mtry,
  min.node.size = best$min.node.size,
  sample.fraction = best$sample.fraction,
  splitrule = "gini",
  probability = TRUE,
  importance = "permutation",
  num.threads = nthreads,
  class.weights = cw,
  save.memory = TRUE,
  seed = 123
)

# Predict (TEST)
rf_test_prob_mat <- predict(rf_final, data = X_test)$predictions
colnames(rf_test_prob_mat) <- levels(y_train)

# Evaluate RF (uses shared helpers)
evaluate_binary <- function(prob_mat, y_true, tag = "model", save_prefix = "outputs") {
  pos_class <- levels(y_true)[1]
  probs <- prob_mat[, pos_class]
  pred_bin <- factor(ifelse(probs >= 0.5, pos_class, setdiff(levels(y_true), pos_class)[1]),
                     levels = levels(y_true))
  cm <- caret::confusionMatrix(pred_bin, y_true, positive = pos_class)
  roc_test <- pROC::roc(y_true == pos_class, probs, quiet = TRUE)
  auc_roc  <- as.numeric(pROC::auc(roc_test))
  cat(sprintf("\n[%s] TEST — Accuracy: %.3f | ROC-AUC: %.3f\n", tag, cm$overall["Accuracy"], auc_roc))
  print(cm)
  plot(roc_test, main=sprintf("%s ROC (TEST) AUC=%.3f", tag, auc_roc))
  if (SAVE_OUTPUTS) { grDevices::dev.copy(png, filename = file.path(save_prefix, paste0(tag, "_ROC_binary.png")), width = 1400, height = 1000, res = 180); dev.off() }
  prb <- pr.curve(scores.class0 = probs[y_true == pos_class], scores.class1 = probs[y_true != pos_class], curve = TRUE)
  plot(prb); if (SAVE_OUTPUTS) { grDevices::dev.copy(png, filename = file.path(save_prefix, paste0(tag, "_PR_binary.png")), width = 1400, height = 1000, res = 180); dev.off() }
  ll <- multiclass_logloss(cbind(neg = 1 - probs, pos = probs)[, levels(y_true)], y_true)
  list(acc = as.numeric(cm$overall["Accuracy"]),
       auc = auc_roc,
       pr_auc = as.numeric(prb$auc.integral),
       logloss = ll)
}
evaluate_multiclass <- function(prob_mat, y_true, X_valid_probs = NULL, y_valid = NULL, tag = "model", save_prefix = "outputs") {
  prob_df <- as.data.frame(prob_mat)
  pred_class_argmax <- factor(colnames(prob_df)[max.col(as.matrix(prob_df))], levels = levels(y_true))
  cm_argmax <- caret::confusionMatrix(pred_class_argmax, y_true)
  per_auc <- c(); first <- TRUE
  for (cls in levels(y_true)) {
    biny <- factor(y_true == cls, levels = c(FALSE, TRUE))
    roc_obj <- try(pROC::roc(biny, prob_df[[cls]], quiet = TRUE), silent = TRUE)
    if (!inherits(roc_obj, "try-error")) {
      if (first) { plot.roc(roc_obj, main = paste0(tag, " OvR ROC (TEST)")); first <- FALSE }
      else { plot.roc(roc_obj, add = TRUE) }
      per_auc[cls] <- as.numeric(pROC::auc(roc_obj))
    }
  }
  if (SAVE_OUTPUTS) { grDevices::dev.copy(png, filename = file.path(save_prefix, paste0(tag, "_ROC_ovr.png")), width = 1400, height = 1000, res = 180); dev.off() }
  macro_auc <- mean(per_auc, na.rm = TRUE)
  pr_auc_vec <- ovr_pr_auc(prob_mat, y_true)
  ll <- multiclass_logloss(prob_mat, y_true)
  f1s <- macro_micro_f1(cm_argmax$table)
  cm_df <- as.data.frame(cm_argmax$table)
  g_cm <- ggplot(cm_df, aes(Reference, Prediction, fill = Freq)) +
    geom_tile() + geom_text(aes(label = Freq), size = 3) +
    theme_minimal() + labs(title = paste0(tag, " — Confusion Matrix (Test, Argmax)"))
  print(g_cm); if (SAVE_OUTPUTS) ggsave(file.path(save_prefix, paste0(tag, "_CM_argmax.png")), g_cm, width = 8, height = 6, dpi = 200)
  out_thresh <- NULL
  if (!is.null(X_valid_probs) && !is.null(y_valid)) {
    t_star <- tune_thresholds_ovr(X_valid_probs, y_valid, grid = seq(0.05, 0.95, by = 0.05))
    cat("\n[", tag, "] Thresholds tuned on VALID:\n", sep = ""); print(round(t_star, 3))
    pred_class_thresh <- predict_with_thresholds(prob_mat, t_star)
    cm_thresh <- caret::confusionMatrix(pred_class_thresh, y_true)
    f1s_thresh <- macro_micro_f1(cm_thresh$table)
    cm_df2 <- as.data.frame(cm_thresh$table)
    g_cm2 <- ggplot(cm_df2, aes(Reference, Prediction, fill = Freq)) +
      geom_tile() + geom_text(aes(label = Freq), size = 3) +
      theme_minimal() + labs(title=paste0(tag, " — Confusion Matrix (Test, Thresholded)"))
    print(g_cm2); if (SAVE_OUTPUTS) ggsave(file.path(save_prefix, paste0(tag, "_CM_thresholded.png")), g_cm2, width = 8, height = 6, dpi = 200)
    out_thresh <- list(accuracy = as.numeric(cm_thresh$overall["Accuracy"]),
                       macro_f1 = f1s_thresh$macro_f1,
                       micro_f1 = f1s_thresh$micro_f1)
  }
  list(
    argmax = list(
      accuracy = as.numeric(cm_argmax$overall["Accuracy"]),
      macro_auc = macro_auc,
      per_class_auc = per_auc,
      pr_auc_ovr = pr_auc_vec,
      macro_f1 = f1s$macro_f1,
      micro_f1 = f1s$micro_f1,
      logloss = ll
    ),
    thresholded = out_thresh
  )
}

# RF evaluation (and valid probs for thresholds)
if (nlevels(y_test) == 2) {
  metrics_list[["rf"]] <- evaluate_binary(rf_test_prob_mat, y_test, tag = "RF")
} else {
  fit_for_valid <- ranger(
    formula = .outcome ~ .,
    data = data.frame(.outcome = y_train, X_train),
    num.trees = best$num.trees %||% 300,
    mtry = best$mtry,
    min.node.size = best$min.node.size,
    sample.fraction = best$sample.fraction,
    splitrule = "gini",
    probability = TRUE,
    num.threads = nthreads,
    class.weights = cw,
    save.memory = TRUE,
    seed = 123
  )
  valid_probs_rf <- predict(fit_for_valid, data = X_valid)$predictions
  colnames(valid_probs_rf) <- levels(y_train)
  metrics_list[["rf"]] <- evaluate_multiclass(rf_test_prob_mat, y_test, X_valid_probs = valid_probs_rf, y_valid = y_valid, tag = "RF")
}

# RF permutation importances
imp <- tibble::enframe(rf_final$variable.importance,
                       name = "feature", value = "importance") %>%
  arrange(desc(importance)) %>% slice_head(n = 15)
g_imp <- ggplot(imp, aes(reorder(feature, importance), importance)) +
  geom_col() + coord_flip() + theme_minimal() +
  labs(title="Random Forest — Top 15 Features (Permutation Importance)", x="", y="Importance")
print(g_imp); if (SAVE_OUTPUTS) ggsave("outputs/08_rf_importance_top15.png", g_imp, width = 8, height = 6, dpi = 200)

# =====================================================================
# ======================  MULTINOMIAL ELASTIC NET  ====================
# =====================================================================
# Sparse matrices from baked data
toSparse <- function(dfm) Matrix::Matrix(as.matrix(dfm), sparse = TRUE)
Xm_train <- toSparse(X_train); Xm_valid <- toSparse(X_valid); Xm_test <- toSparse(X_test)

family_glm <- if (nlevels(y_train) == 2) "binomial" else "multinomial"
type_meas  <- if (family_glm == "binomial") "auc" else "class"

set.seed(123)
cv_fit <- cv.glmnet(
  x = Xm_train,
  y = y_train,
  family = family_glm,
  alpha = 0.5,              # Elastic Net mixture = 0.5
  nfolds = 5,
  type.measure = type_meas,
  parallel = FALSE
)

glmnet_fit <- glmnet(
  x = rbind(Xm_train, Xm_valid),
  y = factor(c(as.character(y_train), as.character(y_valid)), levels = levels(y_train)),
  family = family_glm,
  alpha = 0.5,
  lambda = cv_fit$lambda.min
)

# Predict (TEST)
glm_pred <- predict(glmnet_fit, newx = Xm_test, type = "response")
if (family_glm == "binomial") {
  pos <- levels(y_test)[1]
  prob_mat_glm <- cbind(setNames(1 - as.numeric(glm_pred), levels(y_test)[2]),
                        setNames(as.numeric(glm_pred), pos))
  colnames(prob_mat_glm) <- c(levels(y_test)[2], pos)
  prob_mat_glm <- prob_mat_glm[, levels(y_test)]
  prob_mat_glm <- as.matrix(prob_mat_glm)
} else {
  prob_mat_glm <- as.matrix(glm_pred[,,1])
  colnames(prob_mat_glm) <- levels(y_test)
}

# Evaluate GLMNET (and thresholds using VALID)
if (nlevels(y_test) == 2) {
  metrics_list[["glmnet"]] <- evaluate_binary(prob_mat_glm, y_test, tag = "GLMNET")
} else {
  cv_fit_valid <- cv.glmnet(x = Xm_train, y = y_train, family = "multinomial", alpha = 0.5, nfolds = 5, type.measure = "class")
  glm_valid_fit <- glmnet(x = Xm_train, y = y_train, family = "multinomial", alpha = 0.5, lambda = cv_fit_valid$lambda.min)
  valid_glm_pred <- predict(glm_valid_fit, newx = Xm_valid, type = "response")
  valid_prob_glm <- as.matrix(valid_glm_pred[,,1]); colnames(valid_prob_glm) <- levels(y_train)
  metrics_list[["glmnet"]] <- evaluate_multiclass(prob_mat_glm, y_test, X_valid_probs = valid_prob_glm, y_valid = y_valid, tag = "GLMNET")
}

# =====================================================================
# =======================  GRADIENT-BOOSTED TREES  ====================
# =============================  (XGBoost)  ===========================
# =====================================================================
# Label mapping for xgboost
lbls_map <- setNames(seq_along(levels(y_train)) - 1, levels(y_train))
y_train_int <- unname(lbls_map[as.character(y_train)])
y_valid_int <- unname(lbls_map[as.character(y_valid)])
y_test_int  <- unname(lbls_map[as.character(y_test)])

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train_int)
dvalid <- xgb.DMatrix(data = as.matrix(X_valid), label = y_valid_int)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test_int)

if (nlevels(y_train) == 2) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = c("auc","logloss"),
    eta = if (FAST_MODE) 0.1 else 0.05,
    max_depth = if (FAST_MODE) 5 else 6,
    min_child_weight = 1,
    subsample = 0.7,
    colsample_bytree = 0.7,
    nthread = nthreads
  )
  nrounds <- if (FAST_MODE) 300 else 600
  watchlist <- list(train = dtrain, eval = dvalid)
  set.seed(123)
  xgb_fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = watchlist,
    verbose = 0,
    early_stopping_rounds = if (FAST_MODE) 30 else 50
  )
  xgb_probs_pos <- predict(xgb_fit, dtest)
  pos <- levels(y_test)[1]; neg <- setdiff(levels(y_test), pos)[1]
  xgb_prob_mat <- cbind(neg = 1 - xgb_probs_pos, pos = xgb_probs_pos)
  colnames(xgb_prob_mat) <- c(neg, pos)
  xgb_prob_mat <- xgb_prob_mat[, levels(y_test)]
} else {
  params <- list(
    objective = "multi:softprob",
    num_class = nlevels(y_train),
    eval_metric = c("mlogloss"),
    eta = if (FAST_MODE) 0.1 else 0.05,
    max_depth = if (FAST_MODE) 6 else 7,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = nthreads
  )
  nrounds <- if (FAST_MODE) 400 else 800
  watchlist <- list(train = dtrain, eval = dvalid)
  set.seed(123)
  xgb_fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = watchlist,
    verbose = 0,
    early_stopping_rounds = if (FAST_MODE) 40 else 80
  )
  pred_vec <- predict(xgb_fit, dtest) # length n * num_class
  xgb_prob_mat <- matrix(pred_vec, ncol = nlevels(y_train), byrow = TRUE)
  colnames(xgb_prob_mat) <- levels(y_test)
}

# Evaluate XGB (and thresholds using VALID)
if (nlevels(y_test) == 2) {
  metrics_list[["xgb"]] <- evaluate_binary(xgb_prob_mat, y_test, tag = "XGB")
} else {
  pred_vec_valid <- predict(xgb_fit, dvalid)
  valid_prob_xgb <- matrix(pred_vec_valid, ncol = nlevels(y_train), byrow = TRUE)
  colnames(valid_prob_xgb) <- levels(y_train)
  metrics_list[["xgb"]] <- evaluate_multiclass(xgb_prob_mat, y_test, X_valid_probs = valid_prob_xgb, y_valid = y_valid, tag = "XGB")
}

# XGBoost Importances
xgb_imp <- tryCatch({
  xgb.importance(model = xgb_fit, feature_names = colnames(X_train)) %>% as_tibble() %>% slice_head(n = 15)
}, error = function(e) NULL)
if (!is.null(xgb_imp) && nrow(xgb_imp) > 0) {
  g_ximp <- ggplot(xgb_imp, aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col() + coord_flip() + theme_minimal() +
    labs(title = "XGBoost — Top 15 Features (Gain)", x = "", y = "Gain")
  print(g_ximp); if (SAVE_OUTPUTS) ggsave("outputs/09_xgb_importance_top15.png", g_ximp, width = 8, height = 6, dpi = 200)
}

# ===== 12) Export metrics table =====
if (SAVE_OUTPUTS) {
  flatten_list <- function(lst, prefix = "") {
    out <- list()
    for (nm in names(lst)) {
      val <- lst[[nm]]
      key <- if (nzchar(prefix)) paste0(prefix, ".", nm) else nm
      if (is.list(val)) out <- c(out, Recall = FALSE, flatten_list(val, key))
      else out[[key]] <- val
    }
    out
  }
  flat <- if (length(metrics_list)) flatten_list(metrics_list) else list()
  metrics_tbl <- enframe(flat, name = "metric", value = "value") %>%
    mutate(value = as.character(value))
  readr::write_csv(metrics_tbl, "outputs/10_metrics_summary_all_models.csv")
  cat("\n✅ Saved metrics to outputs/10_metrics_summary_all_models.csv\n")
}

cat("\n🎯 Finished: RF + GLMNET + XGB trained & evaluated.\n")
