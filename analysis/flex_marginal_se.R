# ===========================================================================
# The two greyed columns in analysis/effects_summary.html, with real tests.
#
# The HTML report gives exact POINT estimates for the acoustic and the
# modality-averaged flexibility effects, because both are linear combinations
# of coefficients that lmer does print:
#
#     acoustic        = b[pred:flexX]  +        b[pred:modalityac:flexX]
#     both modalities = b[pred:flexX]  + 0.5  * b[pred:modalityac:flexX]
#
# Their standard errors need cov(b1, b2), which summary(model) suppresses once
# the model has more than 12 fixed effects. This cell recovers them from
# vcov(model). Paste it into a notebook where `model` is still in memory --
# it needs no refit and runs in well under a second.
#
# Prints one row per flexibility moderator per target, matching the report's
# column names, so the two can be checked against each other.
# ===========================================================================

flex_marginal <- function(model, pred = NULL) {
  b  <- lme4::fixef(model)
  V  <- as.matrix(vcov(model))
  nm <- names(b)

  if (is.null(pred)) {
    # the predictor is the term that interacts with modality and nothing else
    cand <- nm[grepl(":modalityac$", nm)]
    pred <- sub(":modalityac$", "", cand[which.min(nchar(cand))])
  }

  est <- function(w) {
    e  <- sum(w * b)
    se <- sqrt(as.numeric(t(w) %*% V %*% w))
    z  <- e / se
    data.frame(estimate = e, SE = se, z = z,
               p = 2 * stats::pnorm(-abs(z)),
               lower = e - 1.96 * se, upper = e + 1.96 * se)
  }

  out <- do.call(rbind, lapply(c("u1_flexibility", "u2_flexibility"), function(f) {
    main <- paste0(pred, ":", f)
    inter <- nm[nm %in% c(paste0(pred, ":modalityac:", f),
                          paste0(pred, ":", f, ":modalityac"))][1]
    stopifnot(main %in% nm, !is.na(inter))
    w0 <- setNames(numeric(length(b)), nm)

    w_art <- w0; w_art[main] <- 1
    w_ac  <- w0; w_ac[main]  <- 1; w_ac[inter]  <- 1
    w_avg <- w0; w_avg[main] <- 1; w_avg[inter] <- 0.5
    w_gap <- w0; w_gap[inter] <- 1

    rbind(cbind(moderator = f, target = "articulatory",   est(w_art)),
          cbind(moderator = f, target = "acoustic",       est(w_ac)),
          cbind(moderator = f, target = "both modalities",est(w_avg)),
          cbind(moderator = f, target = "ac - art",       est(w_gap)))
  }))
  rownames(out) <- NULL
  out
}

res <- flex_marginal(model)
print(res, digits = 4)

# Optional: append to a csv the report can pick up, one row per model.
# write.csv(cbind(dataset = CSV, res),
#           file = "flex_marginal_se.csv", row.names = FALSE, append = FALSE)
