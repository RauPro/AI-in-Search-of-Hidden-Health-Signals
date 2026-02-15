import { z } from "zod";

// ---------------------------------------------------------------------------
// SCALES DOCUMENTATION
// ---------------------------------------------------------------------------
// birth_year (BIRTH_YEAR):   ≥1900, ≤current year
// age:                       auto-computed from birth year (current year − birth year)
// age_squared:               auto-computed (age²)
// gender (RAGENDER):         1=Male, 2=Female  → female = (RAGENDER == 2)
// ethnicity (ETHNICITY):     White | Black | Hispanic | Other
// years_of_education (RAEDYRS): ≥0 numeric
// education_category (RAEDUC):  1–5 select
// highest_degree (RAEDEGRM):    0–6 select
// self_rated_health (SHLT):  1–5 select  (1=Excellent … 5=Poor)
// bmi (BMI):                 10–60 (auto-computed from weight & height, not user-entered)
// weight (WEIGHT):           20–300 kg numeric
// height (HEIGHT):           100–250 cm numeric
// mobility (MOBILA):         0–5 select  (0=No difficulty … 5=Can't do)
// gross_motor (GROSSA):      0–5 select  (same scale)
// large_muscle (LGMUSA):     0–5 select  (same scale)
// fine_motor (FINEA):        0–5 select  (same scale)
// adl (ADL5A):               0–5 select  (# ADL limitations)
// iadl (IADL5A):             0–5 select  (# IADL limitations)
// cognition (COG27):         0–27 numeric (TICS total score)
// memory_recall (TR20):      0–20 numeric (total word recall)
// immediate_recall (IMRC):   0–10 numeric (immediate word recall)
// delayed_recall (DLRC):     0–10 numeric (delayed word recall)
// serial7 (SER7):            0–5 numeric  (serial 7 subtraction)
// cesd (CESD):               0–8 numeric  (CES-D 8-item short form)
// depressed (DEPRES):        0/1 boolean  (felt depressed)
// effort (EFFORT):           0/1 boolean  (everything felt like an effort)
// restless_sleep (SLEEPR):   0/1 boolean  (sleep was restless)
// lonely (FLONE):            0/1 boolean  (felt lonely)
// ever_smoked (SMOKEV):      boolean
// current_smoker (SMOKEN):   boolean (conditional on ever_smoked)
// drinks_per_day (DRINKD):   0–20 numeric
// drink_days_week (DRINKN):  0–7 numeric
// vigorous_activity (VGACTX):0–7 numeric (days per week)
// marital_status (MSTAT):    1–8 select
// condition_count (CONDE):   ≥0 integer
// self_health_comp (SHLTC):  1–3 select (better / same / worse)
// out_of_pocket (OOPMD):     >=0 currency
// working (WORK):            0/1 boolean
// ---------------------------------------------------------------------------

const coerceInt = (min: number, max: number) =>
  z.coerce.number().int().min(min).max(max);

export const healthSchema = z
  .object({
    // Demographics
    BIRTH_YEAR: z.coerce
      .number({ error: "Birth year is required" })
      .int()
      .min(1900, "Must be 1900 or later")
      .max(new Date().getFullYear(), `Must be ${new Date().getFullYear()} or earlier`),
    RAGENDER: z.coerce
      .number({ error: "Gender is required" })
      .int()
      .min(1, "Please select a gender")
      .max(2),
    ETHNICITY: z.enum(["White", "Black", "Hispanic", "Other"], {
      error: "Ethnicity is required",
    }),
    RAEDYRS: z.coerce.number().int().min(0, "Must be 0 or greater"),
    RAEDUC: z.coerce
      .number({ error: "Education category is required" })
      .int()
      .min(1, "Please select a category")
      .max(5),
    RAEDEGRM: z.coerce
      .number({ error: "Highest degree is required" })
      .int()
      .min(0)
      .max(6),

    // General health
    SHLT: z.coerce
      .number({ error: "Self-rated health is required" })
      .int()
      .min(1, "Please select a value between 1 (Excellent) and 5 (Poor)")
      .max(5, "Please select a value between 1 (Excellent) and 5 (Poor)"),

    // Body measures (BMI is auto-computed from weight & height — not in schema)
    WEIGHT: z.coerce
      .number({ error: "Weight is required (20–300 kg)" })
      .min(20, "Min 20 kg")
      .max(300, "Max 300 kg"),
    HEIGHT: z.coerce
      .number({ error: "Height is required (100–250 cm)" })
      .min(100, "Min 100 cm")
      .max(250, "Max 250 cm"),

    // Functional limitations
    MOBILA: coerceInt(0, 5),
    GROSSA: coerceInt(0, 5),
    LGMUSA: coerceInt(0, 5),
    FINEA: coerceInt(0, 5),
    ADL5A: coerceInt(0, 5),
    IADL5A: coerceInt(0, 5),

    // Cognition
    COG27: coerceInt(0, 27),
    TR20: coerceInt(0, 20),
    IMRC: coerceInt(0, 10),
    DLRC: coerceInt(0, 10),
    SER7: coerceInt(0, 5),

    // Depression (CES-D 8-item)
    CESD: coerceInt(0, 8),
    DEPRES: coerceInt(0, 1),
    EFFORT: coerceInt(0, 1),
    SLEEPR: coerceInt(0, 1),
    FLONE: coerceInt(0, 1),

    // Smoking
    SMOKEV: z.boolean(),
    SMOKEN: z.boolean(),

    // Alcohol
    DRINKN: coerceInt(0, 7),
    DRINKD: coerceInt(0, 20),

    // Activity
    VGACTX: coerceInt(0, 7),

    // Sociodemographic
    MSTAT: z.coerce
      .number({ error: "Marital status is required" })
      .int()
      .min(1)
      .max(8),
    CONDE: z.coerce.number({ error: "Number of chronic conditions is required" }).int().min(0, "Must be 0 or greater"),
    SHLTC: z.coerce
      .number({ error: "Health comparison is required" })
      .int()
      .min(1)
      .max(3),

    // Financial / work
    OOPMD: z.coerce.number().min(0, "Must be >= 0"),
    WORK: coerceInt(0, 1),
  })
  .refine(
    (d) => {
      if (d.SMOKEV === false && d.SMOKEN === true) return false;
      return true;
    },
    {
      message: "Cannot be a current smoker without having ever smoked",
      path: ["SMOKEN"],
    },
  );

export type HealthFormValues = z.output<typeof healthSchema>;

// ---------------------------------------------------------------------------
// Per-step field lists — used by the multi-step form to validate each step
// before advancing.
// ---------------------------------------------------------------------------
export const STEP_FIELDS = {
  demographics: [
    "BIRTH_YEAR",
    "RAGENDER",
    "ETHNICITY",
    "RAEDYRS",
    "RAEDUC",
    "RAEDEGRM",
  ] as const,
  generalHealth: ["SHLT", "SHLTC", "CONDE"] as const,
  bodyMeasures: ["WEIGHT", "HEIGHT"] as const,
  functionalLimitations: [
    "MOBILA",
    "GROSSA",
    "LGMUSA",
    "FINEA",
    "ADL5A",
    "IADL5A",
  ] as const,
  cognition: ["COG27", "TR20", "IMRC", "DLRC", "SER7"] as const,
  depression: ["CESD", "DEPRES", "EFFORT", "SLEEPR", "FLONE"] as const,
  lifestyle: ["SMOKEV", "SMOKEN", "DRINKN", "DRINKD", "VGACTX"] as const,
  sociodemographic: ["MSTAT", "WORK", "OOPMD"] as const,
};
