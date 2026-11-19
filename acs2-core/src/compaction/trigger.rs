use std::error::Error;
use std::fmt;

const PERMILLE: u128 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerParameterError {
    ZeroPeriod,
    RatioNotAboveOne,
    YieldOutOfRange,
    InvalidStrengthBounds,
    InvalidStrengthStep,
    InvalidFeedbackParameters,
    InvalidPhaseParameters,
}

impl fmt::Display for TriggerParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::ZeroPeriod => "the period must be positive",
            Self::RatioNotAboveOne => "the growth ratio must exceed 1000 permille",
            Self::YieldOutOfRange => "a yield in permille must not exceed 1000",
            Self::InvalidStrengthBounds => {
                "gentlest strength must not be bolder than boldest strength, and tolerances must be finite and non-negative"
            }
            Self::InvalidStrengthStep => "the tolerance step must be finite and non-negative",
            Self::InvalidFeedbackParameters => {
                "feedback thresholds must satisfy low <= high <= 1000, the restored bound <= 1000 and min <= initial <= max holdoff"
            }
            Self::InvalidPhaseParameters => "the holdoff scale must be at least 1000 permille",
        };
        formatter.write_str(text)
    }
}

impl Error for TriggerParameterError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionStrength {
    pub value_tolerance: f64,
    pub maturity_experience: u32,
    pub idle_audits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrengthStep {
    value_tolerance: f64,
    maturity_experience: u32,
    idle_audits: u32,
}

impl StrengthStep {
    pub fn new(
        value_tolerance: f64,
        maturity_experience: u32,
        idle_audits: u32,
    ) -> Result<Self, TriggerParameterError> {
        if !value_tolerance.is_finite() || value_tolerance < 0.0 {
            return Err(TriggerParameterError::InvalidStrengthStep);
        }
        Ok(Self {
            value_tolerance,
            maturity_experience,
            idle_audits,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrengthBounds {
    gentlest: CompactionStrength,
    boldest: CompactionStrength,
    step: StrengthStep,
}

impl StrengthBounds {
    pub fn new(
        gentlest: CompactionStrength,
        boldest: CompactionStrength,
        step: StrengthStep,
    ) -> Result<Self, TriggerParameterError> {
        let tolerances_valid = gentlest.value_tolerance.is_finite()
            && boldest.value_tolerance.is_finite()
            && gentlest.value_tolerance >= 0.0
            && gentlest.value_tolerance <= boldest.value_tolerance;
        let ordered = gentlest.maturity_experience >= boldest.maturity_experience
            && gentlest.idle_audits >= boldest.idle_audits;
        if !tolerances_valid || !ordered {
            return Err(TriggerParameterError::InvalidStrengthBounds);
        }
        Ok(Self {
            gentlest,
            boldest,
            step,
        })
    }

    pub fn gentlest(&self) -> CompactionStrength {
        self.gentlest
    }

    pub fn boldest(&self) -> CompactionStrength {
        self.boldest
    }

    pub fn clamp(&self, strength: CompactionStrength) -> CompactionStrength {
        CompactionStrength {
            value_tolerance: strength
                .value_tolerance
                .max(self.gentlest.value_tolerance)
                .min(self.boldest.value_tolerance),
            maturity_experience: strength
                .maturity_experience
                .max(self.boldest.maturity_experience)
                .min(self.gentlest.maturity_experience),
            idle_audits: strength
                .idle_audits
                .max(self.boldest.idle_audits)
                .min(self.gentlest.idle_audits),
        }
    }

    pub fn gentler(&self, strength: CompactionStrength, steps: u32) -> CompactionStrength {
        let current = self.clamp(strength);
        CompactionStrength {
            value_tolerance: (current.value_tolerance - self.step.value_tolerance * f64::from(steps))
                .max(self.gentlest.value_tolerance),
            maturity_experience: current
                .maturity_experience
                .saturating_add(self.step.maturity_experience.saturating_mul(steps))
                .min(self.gentlest.maturity_experience),
            idle_audits: current
                .idle_audits
                .saturating_add(self.step.idle_audits.saturating_mul(steps))
                .min(self.gentlest.idle_audits),
        }
    }

    pub fn bolder(&self, strength: CompactionStrength, steps: u32) -> CompactionStrength {
        let current = self.clamp(strength);
        CompactionStrength {
            value_tolerance: (current.value_tolerance + self.step.value_tolerance * f64::from(steps))
                .min(self.boldest.value_tolerance),
            maturity_experience: current
                .maturity_experience
                .saturating_sub(self.step.maturity_experience.saturating_mul(steps))
                .max(self.boldest.maturity_experience),
            idle_audits: current
                .idle_audits
                .saturating_sub(self.step.idle_audits.saturating_mul(steps))
                .max(self.boldest.idle_audits),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriggerObservation {
    pub interactions: u64,
    pub population_size: usize,
    pub match_formations_since_reduction: u64,
    pub classifier_tests_since_reduction: u64,
    pub rules_created_since_reduction: Option<u64>,
    pub coverage_permille: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReductionFeedback {
    pub population_before: usize,
    pub population_after: usize,
    pub work_classifier_tests: u64,
    pub rejected_by_guard: bool,
    pub restored_permille: Option<u32>,
}

impl ReductionFeedback {
    pub fn yield_permille(&self) -> u32 {
        if self.rejected_by_guard || self.population_before == 0 {
            return 0;
        }
        let removed = self.population_before.saturating_sub(self.population_after) as u128;
        (removed * PERMILLE / self.population_before as u128) as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TriggerDecision {
    Skip,
    Compact(CompactionStrength),
}

pub trait TriggerCondition {
    fn is_ready(&self, observation: &TriggerObservation) -> bool;
    fn on_fired(&mut self, observation: &TriggerObservation);
    fn on_reduction(&mut self, feedback: &ReductionFeedback);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Never;

impl TriggerCondition for Never {
    fn is_ready(&self, _observation: &TriggerObservation) -> bool {
        false
    }

    fn on_fired(&mut self, _observation: &TriggerObservation) {}

    fn on_reduction(&mut self, _feedback: &ReductionFeedback) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Once {
    at_interaction: u64,
    fired: bool,
}

impl Once {
    pub fn new(at_interaction: u64) -> Self {
        Self {
            at_interaction,
            fired: false,
        }
    }

    pub fn has_fired(&self) -> bool {
        self.fired
    }
}

impl TriggerCondition for Once {
    fn is_ready(&self, observation: &TriggerObservation) -> bool {
        !self.fired && observation.interactions >= self.at_interaction
    }

    fn on_fired(&mut self, _observation: &TriggerObservation) {
        self.fired = true;
    }

    fn on_reduction(&mut self, _feedback: &ReductionFeedback) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Periodic {
    every_interactions: u64,
    reference_interaction: u64,
}

impl Periodic {
    pub fn new(every_interactions: u64) -> Result<Self, TriggerParameterError> {
        if every_interactions == 0 {
            return Err(TriggerParameterError::ZeroPeriod);
        }
        Ok(Self {
            every_interactions,
            reference_interaction: 0,
        })
    }

    pub fn reference_interaction(&self) -> u64 {
        self.reference_interaction
    }
}

impl TriggerCondition for Periodic {
    fn is_ready(&self, observation: &TriggerObservation) -> bool {
        observation
            .interactions
            .saturating_sub(self.reference_interaction)
            >= self.every_interactions
    }

    fn on_fired(&mut self, observation: &TriggerObservation) {
        self.reference_interaction = observation.interactions;
    }

    fn on_reduction(&mut self, _feedback: &ReductionFeedback) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Growth {
    ratio_permille: u32,
    minimum_reference: usize,
    reference_size: usize,
}

impl Growth {
    pub fn new(
        ratio_permille: u32,
        minimum_reference: usize,
    ) -> Result<Self, TriggerParameterError> {
        if u128::from(ratio_permille) <= PERMILLE {
            return Err(TriggerParameterError::RatioNotAboveOne);
        }
        Ok(Self {
            ratio_permille,
            minimum_reference,
            reference_size: minimum_reference.max(1),
        })
    }

    pub fn reference_size(&self) -> usize {
        self.reference_size
    }
}

impl TriggerCondition for Growth {
    fn is_ready(&self, observation: &TriggerObservation) -> bool {
        observation.population_size as u128 * PERMILLE
            >= self.reference_size as u128 * u128::from(self.ratio_permille)
    }

    fn on_fired(&mut self, observation: &TriggerObservation) {
        self.reference_size = observation.population_size.max(self.minimum_reference).max(1);
    }

    fn on_reduction(&mut self, feedback: &ReductionFeedback) {
        self.reference_size = feedback
            .population_after
            .max(self.minimum_reference)
            .max(1);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostBenefitEstimate {
    pub predicted_savings: u128,
    pub predicted_cost: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PassRecord {
    population_before: usize,
    work_classifier_tests: u64,
    yield_permille: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostBenefit {
    prior_yield_permille: u32,
    yield_floor_permille: u32,
    margin_permille: u32,
    last_pass: Option<PassRecord>,
}

impl CostBenefit {
    pub fn new(
        prior_yield_permille: u32,
        yield_floor_permille: u32,
        margin_permille: u32,
    ) -> Result<Self, TriggerParameterError> {
        if u128::from(prior_yield_permille) > PERMILLE
            || u128::from(yield_floor_permille) > PERMILLE
        {
            return Err(TriggerParameterError::YieldOutOfRange);
        }
        Ok(Self {
            prior_yield_permille,
            yield_floor_permille,
            margin_permille,
            last_pass: None,
        })
    }

    pub fn expected_yield_permille(&self) -> u32 {
        self.last_pass
            .map_or(self.prior_yield_permille, |pass| pass.yield_permille)
            .max(self.yield_floor_permille)
    }

    pub fn estimate(&self, observation: &TriggerObservation) -> CostBenefitEstimate {
        let size = observation.population_size as u128;
        let predicted_savings = u128::from(observation.classifier_tests_since_reduction)
            * u128::from(self.expected_yield_permille())
            / PERMILLE;
        let pairwise = size * size;
        let scaled = match self.last_pass {
            Some(pass) if pass.population_before > 0 => {
                let before = pass.population_before as u128;
                u128::from(pass.work_classifier_tests) * pairwise / (before * before)
            }
            _ => pairwise,
        };
        CostBenefitEstimate {
            predicted_savings,
            predicted_cost: scaled.max(size),
        }
    }

    pub fn last_yield_permille(&self) -> Option<u32> {
        self.last_pass.map(|pass| pass.yield_permille)
    }
}

impl TriggerCondition for CostBenefit {
    fn is_ready(&self, observation: &TriggerObservation) -> bool {
        let estimate = self.estimate(observation);
        u128::from(observation.classifier_tests_since_reduction)
            * u128::from(self.expected_yield_permille())
            > estimate.predicted_cost * u128::from(self.margin_permille)
    }

    fn on_fired(&mut self, _observation: &TriggerObservation) {}

    fn on_reduction(&mut self, feedback: &ReductionFeedback) {
        self.last_pass = Some(PassRecord {
            population_before: feedback.population_before,
            work_classifier_tests: feedback.work_classifier_tests,
            yield_permille: feedback.yield_permille(),
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseTrigger {
    Never(Never),
    Once(Once),
    Periodic(Periodic),
    Growth(Growth),
    CostBenefit(CostBenefit),
}

impl TriggerCondition for BaseTrigger {
    fn is_ready(&self, observation: &TriggerObservation) -> bool {
        match self {
            Self::Never(trigger) => trigger.is_ready(observation),
            Self::Once(trigger) => trigger.is_ready(observation),
            Self::Periodic(trigger) => trigger.is_ready(observation),
            Self::Growth(trigger) => trigger.is_ready(observation),
            Self::CostBenefit(trigger) => trigger.is_ready(observation),
        }
    }

    fn on_fired(&mut self, observation: &TriggerObservation) {
        match self {
            Self::Never(trigger) => trigger.on_fired(observation),
            Self::Once(trigger) => trigger.on_fired(observation),
            Self::Periodic(trigger) => trigger.on_fired(observation),
            Self::Growth(trigger) => trigger.on_fired(observation),
            Self::CostBenefit(trigger) => trigger.on_fired(observation),
        }
    }

    fn on_reduction(&mut self, feedback: &ReductionFeedback) {
        match self {
            Self::Never(trigger) => trigger.on_reduction(feedback),
            Self::Once(trigger) => trigger.on_reduction(feedback),
            Self::Periodic(trigger) => trigger.on_reduction(feedback),
            Self::Growth(trigger) => trigger.on_reduction(feedback),
            Self::CostBenefit(trigger) => trigger.on_reduction(feedback),
        }
    }
}

impl From<Never> for BaseTrigger {
    fn from(trigger: Never) -> Self {
        Self::Never(trigger)
    }
}

impl From<Once> for BaseTrigger {
    fn from(trigger: Once) -> Self {
        Self::Once(trigger)
    }
}

impl From<Periodic> for BaseTrigger {
    fn from(trigger: Periodic) -> Self {
        Self::Periodic(trigger)
    }
}

impl From<Growth> for BaseTrigger {
    fn from(trigger: Growth) -> Self {
        Self::Growth(trigger)
    }
}

impl From<CostBenefit> for BaseTrigger {
    fn from(trigger: CostBenefit) -> Self {
        Self::CostBenefit(trigger)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedbackParameters {
    low_yield_permille: u32,
    high_yield_permille: u32,
    restored_bound_permille: u32,
    minimum_holdoff: u64,
    initial_holdoff: u64,
    maximum_holdoff: u64,
}

impl FeedbackParameters {
    pub fn new(
        low_yield_permille: u32,
        high_yield_permille: u32,
        restored_bound_permille: u32,
        minimum_holdoff: u64,
        initial_holdoff: u64,
        maximum_holdoff: u64,
    ) -> Result<Self, TriggerParameterError> {
        let valid = low_yield_permille <= high_yield_permille
            && u128::from(high_yield_permille) <= PERMILLE
            && u128::from(restored_bound_permille) <= PERMILLE
            && minimum_holdoff <= initial_holdoff
            && initial_holdoff <= maximum_holdoff;
        if !valid {
            return Err(TriggerParameterError::InvalidFeedbackParameters);
        }
        Ok(Self {
            low_yield_permille,
            high_yield_permille,
            restored_bound_permille,
            minimum_holdoff,
            initial_holdoff,
            maximum_holdoff,
        })
    }

    fn lengthened(&self, holdoff: u64) -> u64 {
        holdoff
            .saturating_mul(2)
            .max(self.minimum_holdoff.max(1))
            .clamp(self.minimum_holdoff, self.maximum_holdoff)
    }

    fn shortened(&self, holdoff: u64) -> u64 {
        (holdoff / 2).clamp(self.minimum_holdoff, self.maximum_holdoff)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseModulation {
    creation_rate_ceiling_per_thousand: Option<u64>,
    coverage_floor_permille: Option<u32>,
    strength_steps: u32,
    holdoff_scale_permille: u32,
}

impl PhaseModulation {
    pub fn new(
        creation_rate_ceiling_per_thousand: Option<u64>,
        coverage_floor_permille: Option<u32>,
        strength_steps: u32,
        holdoff_scale_permille: u32,
    ) -> Result<Self, TriggerParameterError> {
        if u128::from(holdoff_scale_permille) < PERMILLE {
            return Err(TriggerParameterError::InvalidPhaseParameters);
        }
        Ok(Self {
            creation_rate_ceiling_per_thousand,
            coverage_floor_permille,
            strength_steps,
            holdoff_scale_permille,
        })
    }

    pub fn is_learning_active(
        &self,
        observation: &TriggerObservation,
        interactions_since_reduction: u64,
    ) -> bool {
        let creation_active = match (
            self.creation_rate_ceiling_per_thousand,
            observation.rules_created_since_reduction,
        ) {
            (Some(ceiling), Some(created)) if interactions_since_reduction > 0 => {
                u128::from(created) * PERMILLE
                    > u128::from(ceiling) * u128::from(interactions_since_reduction)
            }
            _ => false,
        };
        let coverage_active = match (self.coverage_floor_permille, observation.coverage_permille) {
            (Some(floor), Some(coverage)) => coverage < floor,
            _ => false,
        };
        creation_active || coverage_active
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionPolicy {
    trigger: BaseTrigger,
    bounds: StrengthBounds,
    strength: CompactionStrength,
    feedback: Option<FeedbackParameters>,
    phase: Option<PhaseModulation>,
    holdoff_interactions: u64,
    last_fire_interaction: Option<u64>,
    fires: u64,
    reductions: u64,
    last_yield_permille: Option<u32>,
    learning_active: bool,
}

impl CompactionPolicy {
    pub fn new(
        trigger: impl Into<BaseTrigger>,
        bounds: StrengthBounds,
        initial_strength: CompactionStrength,
    ) -> Self {
        Self {
            trigger: trigger.into(),
            bounds,
            strength: bounds.clamp(initial_strength),
            feedback: None,
            phase: None,
            holdoff_interactions: 0,
            last_fire_interaction: None,
            fires: 0,
            reductions: 0,
            last_yield_permille: None,
            learning_active: false,
        }
    }

    pub fn with_feedback(mut self, parameters: FeedbackParameters) -> Self {
        self.holdoff_interactions = parameters.initial_holdoff;
        self.feedback = Some(parameters);
        self
    }

    pub fn with_phase_modulation(mut self, modulation: PhaseModulation) -> Self {
        self.phase = Some(modulation);
        self
    }

    pub fn trigger(&self) -> &BaseTrigger {
        &self.trigger
    }

    pub fn strength(&self) -> CompactionStrength {
        self.strength
    }

    pub fn holdoff_interactions(&self) -> u64 {
        self.holdoff_interactions
    }

    pub fn fires(&self) -> u64 {
        self.fires
    }

    pub fn reductions_reported(&self) -> u64 {
        self.reductions
    }

    pub fn last_yield_permille(&self) -> Option<u32> {
        self.last_yield_permille
    }

    pub fn last_fire_interaction(&self) -> Option<u64> {
        self.last_fire_interaction
    }

    pub fn learning_active(&self) -> bool {
        self.learning_active
    }

    pub fn decide(&mut self, observation: &TriggerObservation) -> TriggerDecision {
        let interactions_since_reduction = self
            .last_fire_interaction
            .map_or(observation.interactions, |fired| {
                observation.interactions.saturating_sub(fired)
            });
        self.learning_active = self.phase.is_some_and(|phase| {
            phase.is_learning_active(observation, interactions_since_reduction)
        });
        if let Some(fired) = self.last_fire_interaction {
            if observation.interactions.saturating_sub(fired) < self.effective_holdoff() {
                return TriggerDecision::Skip;
            }
        }
        if !self.trigger.is_ready(observation) {
            return TriggerDecision::Skip;
        }
        self.trigger.on_fired(observation);
        self.last_fire_interaction = Some(observation.interactions);
        self.fires += 1;
        TriggerDecision::Compact(self.effective_strength())
    }

    pub fn record_reduction(&mut self, feedback: &ReductionFeedback) {
        self.reductions += 1;
        self.trigger.on_reduction(feedback);
        let yield_permille = feedback.yield_permille();
        self.last_yield_permille = Some(yield_permille);
        let Some(parameters) = self.feedback else {
            return;
        };
        let restored = feedback.restored_permille.unwrap_or(0);
        let harmful =
            feedback.rejected_by_guard || restored > parameters.restored_bound_permille;
        self.strength = if harmful {
            self.bounds.gentler(self.strength, 1)
        } else {
            self.bounds.bolder(self.strength, 1)
        };
        if yield_permille < parameters.low_yield_permille {
            self.holdoff_interactions = parameters.lengthened(self.holdoff_interactions);
        } else if !harmful && yield_permille >= parameters.high_yield_permille {
            self.holdoff_interactions = parameters.shortened(self.holdoff_interactions);
        }
    }

    fn effective_holdoff(&self) -> u64 {
        match self.phase {
            Some(phase) if self.learning_active => {
                let scaled = u128::from(self.holdoff_interactions)
                    * u128::from(phase.holdoff_scale_permille)
                    / PERMILLE;
                u64::try_from(scaled).unwrap_or(u64::MAX)
            }
            _ => self.holdoff_interactions,
        }
    }

    fn effective_strength(&self) -> CompactionStrength {
        match self.phase {
            Some(phase) if self.learning_active => {
                self.bounds.gentler(self.strength, phase.strength_steps)
            }
            _ => self.strength,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strength(tolerance: f64, maturity: u32, audits: u32) -> CompactionStrength {
        CompactionStrength {
            value_tolerance: tolerance,
            maturity_experience: maturity,
            idle_audits: audits,
        }
    }

    fn bounds() -> StrengthBounds {
        StrengthBounds::new(
            strength(0.0, 20, 6),
            strength(0.5, 4, 1),
            StrengthStep::new(0.125, 4, 1).unwrap(),
        )
        .unwrap()
    }

    fn middle() -> CompactionStrength {
        strength(0.25, 12, 3)
    }

    fn at(interactions: u64, population: usize) -> TriggerObservation {
        TriggerObservation {
            interactions,
            population_size: population,
            ..TriggerObservation::default()
        }
    }

    fn with_work(
        interactions: u64,
        population: usize,
        formations: u64,
        tests: u64,
    ) -> TriggerObservation {
        TriggerObservation {
            interactions,
            population_size: population,
            match_formations_since_reduction: formations,
            classifier_tests_since_reduction: tests,
            ..TriggerObservation::default()
        }
    }

    fn feedback(before: usize, after: usize, work: u64) -> ReductionFeedback {
        ReductionFeedback {
            population_before: before,
            population_after: after,
            work_classifier_tests: work,
            rejected_by_guard: false,
            restored_permille: None,
        }
    }

    fn policy(trigger: impl Into<BaseTrigger>) -> CompactionPolicy {
        CompactionPolicy::new(trigger, bounds(), middle())
    }

    fn fired(decision: TriggerDecision) -> bool {
        matches!(decision, TriggerDecision::Compact(_))
    }

    #[test]
    fn never_does_not_fire() {
        let mut policy = policy(Never);
        for step in 1..50 {
            let observation = with_work(step * 1000, 10_000, 1_000_000, u64::MAX);
            assert_eq!(policy.decide(&observation), TriggerDecision::Skip);
        }
        assert_eq!(policy.fires(), 0);
    }

    #[test]
    fn once_fires_exactly_once_at_first_boundary_at_or_after() {
        let mut policy = policy(Once::new(1000));
        assert!(!fired(policy.decide(&at(500, 100))));
        assert!(!fired(policy.decide(&at(999, 100))));
        assert!(fired(policy.decide(&at(1040, 100))));
        policy.record_reduction(&feedback(100, 60, 500));
        for step in 0..20 {
            assert!(!fired(policy.decide(&at(1100 + step * 1000, 100_000))));
        }
        assert_eq!(policy.fires(), 1);
        assert!(matches!(policy.trigger(), BaseTrigger::Once(once) if once.has_fired()));
    }

    #[test]
    fn once_fires_on_exact_interaction() {
        let mut policy = policy(Once::new(1000));
        assert!(fired(policy.decide(&at(1000, 5))));
    }

    #[test]
    fn periodic_fires_every_period_from_the_last_fire() {
        let mut policy = policy(Periodic::new(100).unwrap());
        let schedule: Vec<bool> = [50, 99, 100, 150, 199, 200, 290, 300, 410, 500]
            .iter()
            .map(|interactions| fired(policy.decide(&at(*interactions, 10))))
            .collect();
        assert_eq!(
            schedule,
            [false, false, true, false, false, true, false, true, true, false]
        );
    }

    #[test]
    fn periodic_rejects_zero_period() {
        assert_eq!(Periodic::new(0), Err(TriggerParameterError::ZeroPeriod));
    }

    #[test]
    fn growth_fires_at_ratio_over_initial_reference() {
        let mut policy = policy(Growth::new(1500, 40).unwrap());
        assert!(!fired(policy.decide(&at(10, 59))));
        assert!(fired(policy.decide(&at(20, 60))));
    }

    #[test]
    fn growth_resets_reference_to_size_after_reduction() {
        let mut policy = policy(Growth::new(1500, 20).unwrap());
        assert!(fired(policy.decide(&at(10, 100))));
        policy.record_reduction(&feedback(100, 50, 1000));
        assert!(matches!(policy.trigger(), BaseTrigger::Growth(growth) if growth.reference_size() == 50));
        assert!(!fired(policy.decide(&at(20, 74))));
        assert!(fired(policy.decide(&at(30, 75))));
        policy.record_reduction(&feedback(75, 30, 1000));
        assert!(matches!(policy.trigger(), BaseTrigger::Growth(growth) if growth.reference_size() == 30));
        assert!(!fired(policy.decide(&at(40, 44))));
        assert!(fired(policy.decide(&at(50, 45))));
    }

    #[test]
    fn growth_reference_never_falls_below_minimum() {
        let mut policy = policy(Growth::new(2000, 64).unwrap());
        assert!(fired(policy.decide(&at(10, 200))));
        policy.record_reduction(&feedback(200, 5, 1000));
        assert!(!fired(policy.decide(&at(20, 127))));
        assert!(fired(policy.decide(&at(30, 128))));
    }

    #[test]
    fn growth_rejects_ratio_not_above_one() {
        assert_eq!(Growth::new(1000, 10), Err(TriggerParameterError::RatioNotAboveOne));
        assert_eq!(Growth::new(900, 10), Err(TriggerParameterError::RatioNotAboveOne));
    }

    #[test]
    fn cost_benefit_uses_pairwise_estimate_and_prior_before_first_pass() {
        let trigger = CostBenefit::new(100, 10, 1000).unwrap();
        let mut policy = policy(trigger);
        let estimate = match policy.trigger() {
            BaseTrigger::CostBenefit(trigger) => trigger.estimate(&with_work(0, 100, 0, 100_000)),
            _ => unreachable!(),
        };
        assert_eq!(estimate.predicted_cost, 10_000);
        assert_eq!(estimate.predicted_savings, 10_000);
        assert!(!fired(policy.decide(&with_work(1000, 100, 1000, 100_000))));
        assert!(fired(policy.decide(&with_work(1001, 100, 1001, 100_001))));
    }

    #[test]
    fn cost_benefit_scales_last_pass_work_quadratically_with_size() {
        let mut policy = policy(CostBenefit::new(100, 10, 1000).unwrap());
        assert!(fired(policy.decide(&with_work(1000, 100, 10, 1_000_000))));
        policy.record_reduction(&feedback(100, 50, 4000));
        assert!(!fired(policy.decide(&with_work(2000, 60, 10, 2880))));
        assert!(fired(policy.decide(&with_work(2001, 60, 10, 2882))));
    }

    #[test]
    fn cost_benefit_margin_demands_larger_savings() {
        let strict = CostBenefit::new(100, 10, 2000).unwrap();
        let mut policy = policy(strict);
        assert!(!fired(policy.decide(&with_work(1000, 100, 1, 200_000))));
        assert!(fired(policy.decide(&with_work(1001, 100, 1, 200_001))));
    }

    #[test]
    fn cost_benefit_yield_floor_backs_off_after_empty_pass_without_deadlock() {
        let mut policy = policy(CostBenefit::new(100, 10, 1000).unwrap());
        assert!(fired(policy.decide(&with_work(1000, 100, 1, 1_000_000))));
        policy.record_reduction(&feedback(100, 100, 10_000));
        assert!(!fired(policy.decide(&with_work(2000, 100, 1, 999_999))));
        assert!(fired(policy.decide(&with_work(2001, 100, 1, 1_000_001))));
    }

    #[test]
    fn cost_benefit_treats_guard_rejection_as_zero_yield() {
        let mut trigger = CostBenefit::new(100, 10, 1000).unwrap();
        trigger.on_reduction(&ReductionFeedback {
            rejected_by_guard: true,
            ..feedback(100, 40, 800)
        });
        assert_eq!(trigger.last_yield_permille(), Some(0));
    }

    #[test]
    fn cost_benefit_never_fires_on_empty_population() {
        let policy_trigger = CostBenefit::new(100, 10, 1000).unwrap();
        assert!(!policy_trigger.is_ready(&with_work(10, 0, 0, 0)));
    }

    #[test]
    fn cost_benefit_rejects_yield_above_one() {
        assert_eq!(
            CostBenefit::new(1001, 10, 1000),
            Err(TriggerParameterError::YieldOutOfRange)
        );
        assert_eq!(
            CostBenefit::new(100, 1001, 1000),
            Err(TriggerParameterError::YieldOutOfRange)
        );
    }

    fn controlled() -> CompactionPolicy {
        let parameters = FeedbackParameters::new(100, 400, 200, 100, 100, 1600).unwrap();
        policy(Periodic::new(100).unwrap()).with_feedback(parameters)
    }

    fn fire_and_report(policy: &mut CompactionPolicy, at_interaction: u64, report: ReductionFeedback) {
        assert!(fired(policy.decide(&at(at_interaction, 100))));
        policy.record_reduction(&report);
    }

    #[test]
    fn controller_gets_gentler_after_guard_rejection() {
        let mut policy = controlled();
        fire_and_report(
            &mut policy,
            100,
            ReductionFeedback {
                rejected_by_guard: true,
                ..feedback(100, 100, 500)
            },
        );
        assert_eq!(policy.strength(), strength(0.125, 16, 4));
    }

    #[test]
    fn controller_gets_gentler_when_restored_fraction_exceeds_bound() {
        let mut policy = controlled();
        fire_and_report(
            &mut policy,
            100,
            ReductionFeedback {
                restored_permille: Some(201),
                ..feedback(100, 40, 500)
            },
        );
        assert_eq!(policy.strength(), strength(0.125, 16, 4));
    }

    #[test]
    fn controller_keeps_bound_restoration_as_clean() {
        let mut policy = controlled();
        fire_and_report(
            &mut policy,
            100,
            ReductionFeedback {
                restored_permille: Some(200),
                ..feedback(100, 40, 500)
            },
        );
        assert_eq!(policy.strength(), strength(0.375, 8, 2));
    }

    #[test]
    fn controller_gets_bolder_after_clean_high_yield_and_shortens_interval() {
        let mut policy = controlled();
        fire_and_report(&mut policy, 100, feedback(100, 50, 500));
        assert_eq!(policy.strength(), strength(0.375, 8, 2));
        assert_eq!(policy.holdoff_interactions(), 100);
        assert!(!fired(policy.decide(&at(150, 100))));
        policy.holdoff_interactions = 400;
        fire_and_report(&mut policy, 600, feedback(100, 50, 500));
        assert_eq!(policy.holdoff_interactions(), 200);
    }

    #[test]
    fn controller_lengthens_interval_after_low_yield() {
        let mut policy = controlled();
        fire_and_report(&mut policy, 100, feedback(100, 95, 500));
        assert_eq!(policy.holdoff_interactions(), 200);
        assert!(!fired(policy.decide(&at(299, 100))));
        fire_and_report(&mut policy, 300, feedback(100, 100, 500));
        assert_eq!(policy.holdoff_interactions(), 400);
        assert!(!fired(policy.decide(&at(699, 100))));
        assert!(fired(policy.decide(&at(700, 100))));
    }

    #[test]
    fn controller_holdoff_is_clamped_to_maximum() {
        let mut policy = controlled();
        let mut now = 0;
        for _ in 0..8 {
            now += 10_000;
            fire_and_report(&mut policy, now, feedback(100, 100, 500));
        }
        assert_eq!(policy.holdoff_interactions(), 1600);
    }

    #[test]
    fn controller_holdoff_is_clamped_to_minimum() {
        let mut policy = controlled();
        let mut now = 0;
        for _ in 0..5 {
            now += 10_000;
            fire_and_report(&mut policy, now, feedback(100, 10, 500));
        }
        assert_eq!(policy.holdoff_interactions(), 100);
    }

    #[test]
    fn controller_leaves_interval_on_middle_yield() {
        let mut policy = controlled();
        fire_and_report(&mut policy, 100, feedback(100, 80, 500));
        assert_eq!(policy.holdoff_interactions(), 100);
    }

    #[test]
    fn strength_is_clamped_to_bounds_in_both_directions() {
        let mut policy = controlled();
        let mut now = 0;
        for _ in 0..10 {
            now += 10_000;
            fire_and_report(&mut policy, now, feedback(100, 50, 500));
        }
        assert_eq!(policy.strength(), bounds().boldest());
        for _ in 0..10 {
            now += 10_000;
            fire_and_report(
                &mut policy,
                now,
                ReductionFeedback {
                    rejected_by_guard: true,
                    ..feedback(100, 100, 500)
                },
            );
        }
        assert_eq!(policy.strength(), bounds().gentlest());
    }

    #[test]
    fn decision_carries_the_current_strength() {
        let mut policy = controlled();
        assert_eq!(
            policy.decide(&at(100, 100)),
            TriggerDecision::Compact(middle())
        );
    }

    #[test]
    fn policy_without_controller_keeps_strength_fixed() {
        let mut policy = policy(Periodic::new(100).unwrap());
        fire_and_report(
            &mut policy,
            100,
            ReductionFeedback {
                rejected_by_guard: true,
                ..feedback(100, 100, 500)
            },
        );
        assert_eq!(policy.strength(), middle());
        assert_eq!(policy.holdoff_interactions(), 0);
        assert_eq!(policy.last_yield_permille(), Some(0));
    }

    #[test]
    fn initial_strength_is_clamped_to_bounds() {
        let policy = CompactionPolicy::new(Never, bounds(), strength(9.0, 0, 0));
        assert_eq!(policy.strength(), bounds().boldest());
    }

    #[test]
    fn bounds_reject_inverted_ordering() {
        let step = StrengthStep::new(0.1, 1, 1).unwrap();
        assert_eq!(
            StrengthBounds::new(strength(0.5, 20, 6), strength(0.1, 4, 1), step),
            Err(TriggerParameterError::InvalidStrengthBounds)
        );
        assert_eq!(
            StrengthBounds::new(strength(0.0, 4, 6), strength(0.5, 20, 1), step),
            Err(TriggerParameterError::InvalidStrengthBounds)
        );
        assert_eq!(
            StrengthStep::new(f64::NAN, 1, 1),
            Err(TriggerParameterError::InvalidStrengthStep)
        );
    }

    #[test]
    fn feedback_parameters_reject_inconsistent_values() {
        assert!(FeedbackParameters::new(500, 400, 200, 1, 1, 8).is_err());
        assert!(FeedbackParameters::new(100, 400, 1001, 1, 1, 8).is_err());
        assert!(FeedbackParameters::new(100, 400, 200, 4, 2, 8).is_err());
        assert!(FeedbackParameters::new(100, 400, 200, 1, 9, 8).is_err());
    }

    #[test]
    fn yield_is_removed_fraction_in_permille() {
        assert_eq!(feedback(200, 150, 0).yield_permille(), 250);
        assert_eq!(feedback(0, 0, 0).yield_permille(), 0);
        assert_eq!(feedback(10, 12, 0).yield_permille(), 0);
    }

    fn modulated(trigger: impl Into<BaseTrigger>) -> CompactionPolicy {
        let modulation = PhaseModulation::new(Some(100), Some(800), 2, 2000).unwrap();
        policy(trigger).with_phase_modulation(modulation)
    }

    #[test]
    fn phase_modulation_gentles_strength_when_rule_creation_is_high() {
        let mut policy = modulated(Periodic::new(100).unwrap());
        let observation = TriggerObservation {
            rules_created_since_reduction: Some(11),
            ..at(100, 50)
        };
        assert_eq!(
            policy.decide(&observation),
            TriggerDecision::Compact(strength(0.0, 20, 5))
        );
        assert!(policy.learning_active());
        assert_eq!(policy.strength(), middle());
    }

    #[test]
    fn phase_modulation_gentles_strength_when_coverage_is_low() {
        let mut policy = modulated(Periodic::new(100).unwrap());
        let observation = TriggerObservation {
            coverage_permille: Some(799),
            ..at(100, 50)
        };
        assert_eq!(
            policy.decide(&observation),
            TriggerDecision::Compact(strength(0.0, 20, 5))
        );
    }

    #[test]
    fn phase_modulation_is_inert_when_learning_is_quiet_or_signals_missing() {
        let mut policy = modulated(Periodic::new(100).unwrap());
        let quiet = TriggerObservation {
            rules_created_since_reduction: Some(10),
            coverage_permille: Some(800),
            ..at(100, 50)
        };
        assert_eq!(policy.decide(&quiet), TriggerDecision::Compact(middle()));
        assert!(!policy.learning_active());
        assert_eq!(
            policy.decide(&at(200, 50)),
            TriggerDecision::Compact(middle())
        );
    }

    #[test]
    fn phase_modulation_scales_the_holdoff_but_never_forces_a_fire() {
        let parameters = FeedbackParameters::new(0, 1000, 1000, 100, 100, 1600).unwrap();
        let mut policy = modulated(Periodic::new(100).unwrap()).with_feedback(parameters);
        assert!(fired(policy.decide(&at(100, 50))));
        policy.record_reduction(&feedback(50, 25, 100));
        let active = TriggerObservation {
            coverage_permille: Some(0),
            ..at(250, 50)
        };
        assert_eq!(policy.decide(&active), TriggerDecision::Skip);
        let active_late = TriggerObservation {
            coverage_permille: Some(0),
            ..at(300, 50)
        };
        assert!(fired(policy.decide(&active_late)));

        let mut never = modulated(Never);
        let urgent = TriggerObservation {
            coverage_permille: Some(0),
            rules_created_since_reduction: Some(1_000_000),
            ..at(1_000_000, 1_000_000)
        };
        assert_eq!(never.decide(&urgent), TriggerDecision::Skip);
    }

    #[test]
    fn phase_modulation_clamps_to_gentlest_strength() {
        let modulation = PhaseModulation::new(None, Some(1000), 100, 1000).unwrap();
        let mut policy = policy(Periodic::new(10).unwrap()).with_phase_modulation(modulation);
        let observation = TriggerObservation {
            coverage_permille: Some(0),
            ..at(10, 50)
        };
        assert_eq!(
            policy.decide(&observation),
            TriggerDecision::Compact(bounds().gentlest())
        );
    }

    #[test]
    fn phase_modulation_rejects_scale_below_one() {
        assert_eq!(
            PhaseModulation::new(None, None, 1, 999),
            Err(TriggerParameterError::InvalidPhaseParameters)
        );
    }

    #[test]
    fn identical_sequences_give_identical_decisions_and_state() {
        let parameters = FeedbackParameters::new(100, 400, 200, 100, 100, 1600).unwrap();
        let modulation = PhaseModulation::new(Some(50), Some(900), 1, 1500).unwrap();
        let build = || {
            policy(CostBenefit::new(500, 10, 1000).unwrap())
                .with_feedback(parameters)
                .with_phase_modulation(modulation)
        };
        let run = |policy: &mut CompactionPolicy| {
            let mut decisions = Vec::new();
            let mut size = 80_usize;
            for step in 1..200_u64 {
                size += 7;
                let observation = TriggerObservation {
                    interactions: step * 137,
                    population_size: size,
                    match_formations_since_reduction: step * 10,
                    classifier_tests_since_reduction: step * 100 * size as u64,
                    rules_created_since_reduction: Some(step * 3),
                    coverage_permille: Some((step * 11 % 1000) as u32),
                };
                let decision = policy.decide(&observation);
                if fired(decision) {
                    let after = size / 2;
                    policy.record_reduction(&ReductionFeedback {
                        population_before: size,
                        population_after: after,
                        work_classifier_tests: (size * size / 2) as u64,
                        rejected_by_guard: step % 17 == 0,
                        restored_permille: Some((step * 7 % 400) as u32),
                    });
                    size = after;
                }
                decisions.push(decision);
            }
            decisions
        };
        let mut first = build();
        let mut second = build();
        let first_decisions = run(&mut first);
        let second_decisions = run(&mut second);
        assert_eq!(first_decisions, second_decisions);
        assert_eq!(first, second);
        assert!(first.fires() > 1);
        assert_eq!(first.clone(), first);
    }
}
