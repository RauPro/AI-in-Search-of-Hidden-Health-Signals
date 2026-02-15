import type { HealthFormValues } from "./schema";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function computeBMI(weightKg: number, heightCm: number): number {
  const heightM = heightCm / 100;
  return Math.round((weightKg / (heightM * heightM)) * 10) / 10;
}

export function computeAge(birthYear: number): number {
  return new Date().getFullYear() - birthYear;
}

// ---------------------------------------------------------------------------
// API payload
// ---------------------------------------------------------------------------

export type ApiPayload = {
  age: number;
  female: number;
  ethnicity: string;
  height: number;
  weight: number;
  bmi: number;
  bmi_lag1: number;
  self_rated_health: number;
  self_rated_health_lag1: number;
  mobility: number;
  gross_motor: number;
  fine_motor: number;
  large_muscle: number;
  adl: number;
  iadl: number;
  cognition: number;
  memory_recall: number;
  serial7: number;
  cesd: number;
  depressed: number;
  lonely: number;
  restless_sleep: number;
  effort: number;
  ever_smoked: number;
  current_smoker: number;
  drinks_per_day: number;
  drink_days_week: number;
  vigorous_activity: number;
  working: number;
};

export function buildApiPayload(
  data: HealthFormValues,
  computedAge: number | null,
  computedBMI: number | null,
): ApiPayload {
  return {
    age: computedAge ?? 0,
    female: data.RAGENDER === 2 ? 1 : 0,
    ethnicity: data.ETHNICITY,
    height: data.HEIGHT / 100, // cm -> meters
    weight: data.WEIGHT,
    bmi: computedBMI ?? 0,
    bmi_lag1: 0, // not collected, default 0
    self_rated_health: data.SHLT,
    self_rated_health_lag1: 0, // not collected, default 0
    mobility: data.MOBILA,
    gross_motor: data.GROSSA,
    fine_motor: data.FINEA,
    large_muscle: data.LGMUSA,
    adl: data.ADL5A,
    iadl: data.IADL5A,
    cognition: data.COG27,
    memory_recall: data.TR20,
    serial7: data.SER7,
    cesd: data.CESD,
    depressed: data.DEPRES,
    lonely: data.FLONE,
    restless_sleep: data.SLEEPR,
    effort: data.EFFORT,
    ever_smoked: data.SMOKEV ? 1 : 0, // boolean -> 0/1
    current_smoker: data.SMOKEN ? 1 : 0, // boolean -> 0/1
    drinks_per_day: data.DRINKD,
    drink_days_week: data.DRINKN,
    vigorous_activity: data.VGACTX,
    working: data.WORK,
  };
}

// ---------------------------------------------------------------------------
// Option arrays (feature-specific, used across multiple steps)
// ---------------------------------------------------------------------------

export const genderOptions = [
  { value: 1, label: "Male" },
  { value: 2, label: "Female" },
];

export const ethnicityOptions = [
  { value: "White", label: "White" },
  { value: "Black", label: "Black" },
  { value: "Hispanic", label: "Hispanic" },
  { value: "Other", label: "Other" },
];

export const educationCategoryOptions = [
  { value: 1, label: "1 – Less than high school" },
  { value: 2, label: "2 – General Educational Development" },
  { value: 3, label: "3 – High school graduate" },
  { value: 4, label: "4 – Some college" },
  { value: 5, label: "5 – College and above" },
];

export const degreeOptions = [
  { value: 0, label: "0 – No degree" },
  { value: 1, label: "1 – General Educational Development" },
  { value: 2, label: "2 – High school diploma" },
  { value: 3, label: "3 – Two year college degree" },
  { value: 4, label: "4 – Four year college degree" },
  { value: 5, label: "5 – Master degree" },
  { value: 6, label: "6 – Professional degree (PhD, MD, JD)" },
];

export const difficultyOptions = [
  { value: 0, label: "No trouble at all" },
  { value: 1, label: "A little difficult" },
  { value: 2, label: "Quite difficult" },
  { value: 3, label: "Unable to do" },
  { value: 4, label: "Choose not to do" },
  { value: 5, label: "Not applicable" },
];

export const selfRatedOptions = [
  { value: 1, label: "1 – Excellent" },
  { value: 2, label: "2 – Very Good" },
  { value: 3, label: "3 – Good" },
  { value: 4, label: "4 – Fair" },
  { value: 5, label: "5 – Poor" },
];

export const maritalOptions = [
  { value: 1, label: "1 – Married" },
  { value: 2, label: "2 – Married, spouse absent" },
  { value: 3, label: "3 – Partnered" },
  { value: 4, label: "4 – Separated" },
  { value: 5, label: "5 – Divorced" },
  { value: 6, label: "6 – Widowed" },
  { value: 7, label: "7 – Never married" },
  { value: 8, label: "8 – Unknown" },
];

export const healthCompOptions = [
  { value: 1, label: "1 – Better" },
  { value: 2, label: "2 – About the same" },
  { value: 3, label: "3 – Worse" },
];

export const yesNoOptions = [
  { value: 0, label: "No" },
  { value: 1, label: "Yes" },
];

// ---------------------------------------------------------------------------
// Cognition mini-game word pools (CERAD-style 10-word lists)
// ---------------------------------------------------------------------------

export const WORD_POOLS = [
  ["butter", "arm", "shore", "letter", "queen", "cabin", "pole", "ticket", "grass", "engine"],
  ["hotel", "river", "tree", "skin", "gold", "market", "paper", "child", "king", "book"],
  ["finger", "garden", "wagon", "church", "fish", "hammer", "stone", "cloud", "bell", "train"],
];
