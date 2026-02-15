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
  { value: 0, label: "0 – No difficulty" },
  { value: 1, label: "1 – Some difficulty" },
  { value: 2, label: "2 – A lot of difficulty" },
  { value: 3, label: "3 – Cannot do" },
  { value: 4, label: "4 – Don't do" },
  { value: 5, label: "5 – N/A" },
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
