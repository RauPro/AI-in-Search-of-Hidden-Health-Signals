"use client";

import { useEffect } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

// ---------------------------------------------------------------------------
// SCALES DOCUMENTATION
// ---------------------------------------------------------------------------
// self_rated_health (SHLT):  1–5 select  (1=Excellent … 5=Poor)
// bmi (BMI):                 10–60 numeric, auto-computed when empty
// weight (WEIGHT):           30–250 kg numeric
// height (HEIGHT):           120–220 cm numeric
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
// condition_count (CONDE):   0–10 integer
// self_health_comp (SHLTC):  1–3 select (better / same / worse)
// out_of_pocket (OOPMD):     >=0 currency
// working (WORK):            0/1 boolean
// ---------------------------------------------------------------------------

// ---- Zod Schema -----------------------------------------------------------

const coerceInt = (min: number, max: number) =>
  z.coerce.number().int().min(min).max(max);

const healthSchema = z
  .object({
    // General health
    SHLT: z.coerce
      .number({ error: "Self-rated health is required (1–5)" })
      .int()
      .min(1, "Min 1")
      .max(5, "Max 5"),

    // Body measures – BMI accepts empty string for auto-compute
    BMI: z.union([z.literal(""), z.coerce.number().min(10).max(60)]).optional(),
    WEIGHT: z.coerce
      .number({ error: "Weight is required (30–250 kg)" })
      .min(30, "Min 30 kg")
      .max(250, "Max 250 kg"),
    HEIGHT: z.coerce
      .number({ error: "Height is required (120–220 cm)" })
      .min(120, "Min 120 cm")
      .max(220, "Max 220 cm"),

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
    MSTAT: z.coerce.number({ error: "Marital status is required" }).int().min(1).max(8),
    CONDE: coerceInt(0, 10),
    SHLTC: z.coerce.number({ error: "Health comparison is required" }).int().min(1).max(3),

    // Financial / work
    OOPMD: z.coerce.number().min(0, "Must be >= 0"),
    WORK: coerceInt(0, 1),
  })
  .refine(
    (d) => {
      if (d.SMOKEV === false && d.SMOKEN === true) return false;
      return true;
    },
    { message: "Cannot be a current smoker without having ever smoked", path: ["SMOKEN"] }
  );

type HealthFormValues = z.output<typeof healthSchema>;

// ---- Stub submit ----------------------------------------------------------

function submitAssessment(payload: HealthFormValues) {
  // TODO: wire to API
  console.log("submitAssessment payload:", payload);
}

// ---- Helpers --------------------------------------------------------------

function computeBMI(weightKg: number, heightCm: number): number {
  const heightM = heightCm / 100;
  return Math.round((weightKg / (heightM * heightM)) * 10) / 10;
}

// ---- Reusable tiny components ---------------------------------------------

function FieldWrapper({
  id,
  label,
  helper,
  error,
  children,
}: {
  id: string;
  label: string;
  helper?: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-semibold text-gray-700">
        {label}
      </label>
      {helper && (
        <span className="text-xs text-gray-500" id={`${id}-helper`}>
          {helper}
        </span>
      )}
      {children}
      {error && (
        <span role="alert" className="text-xs text-red-600" id={`${id}-error`}>
          {error}
        </span>
      )}
    </div>
  );
}

const inputBase =
  "w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm transition focus:border-green-mid focus:outline-none focus:ring-2 focus:ring-green-mid/40 disabled:bg-gray-100 disabled:text-gray-500";

const selectBase =
  "w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm transition focus:border-green-mid focus:outline-none focus:ring-2 focus:ring-green-mid/40";

// ---- MAIN COMPONENT ------------------------------------------------------

export default function HealthIntakeForm() {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    control,
    formState: { errors },
  } = useForm<HealthFormValues>({
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    resolver: zodResolver(healthSchema) as any,
    defaultValues: {
      SHLT: undefined,
      BMI: "",
      WEIGHT: undefined,
      HEIGHT: undefined,
      MOBILA: 0,
      GROSSA: 0,
      LGMUSA: 0,
      FINEA: 0,
      ADL5A: 0,
      IADL5A: 0,
      COG27: 0,
      TR20: 0,
      IMRC: 0,
      DLRC: 0,
      SER7: 0,
      CESD: 0,
      DEPRES: 0,
      EFFORT: 0,
      SLEEPR: 0,
      FLONE: 0,
      SMOKEV: false,
      SMOKEN: false,
      DRINKN: 0,
      DRINKD: 0,
      VGACTX: 0,
      MSTAT: undefined,
      CONDE: 0,
      SHLTC: undefined,
      OOPMD: 0,
      WORK: 0,
    },
  });

  // -- Watchers for conditional logic --
  const everSmoked = watch("SMOKEV");
  const drinkDays = watch("DRINKN");
  const weight = watch("WEIGHT");
  const height = watch("HEIGHT");
  const bmiRaw = watch("BMI");

  // Hide current_smoker when ever_smoked is false
  useEffect(() => {
    if (!everSmoked) setValue("SMOKEN", false);
  }, [everSmoked, setValue]);

  // Force drinks_per_day to 0 when drink_days_week == 0
  useEffect(() => {
    if (drinkDays === 0) setValue("DRINKD", 0);
  }, [drinkDays, setValue]);

  // Auto-compute BMI
  const computedBMI =
    weight && height && weight >= 30 && height >= 120
      ? computeBMI(weight, height)
      : null;

  const showComputedBMI = (bmiRaw === "" || bmiRaw === undefined) && computedBMI !== null;

  // -- Submit handler --
  const onSubmit = (data: HealthFormValues) => {
    const payload = {
      ...data,
      BMI: showComputedBMI ? computedBMI : data.BMI,
    };
    console.log("Form payload:", payload);
    submitAssessment(payload as HealthFormValues);
  };

  // -- Error helper --
  function err(field: keyof HealthFormValues): string | undefined {
    const e = errors[field];
    return e?.message as string | undefined;
  }

  // -- Option builders --
  const difficultyOptions = [
    { value: 0, label: "0 – No difficulty" },
    { value: 1, label: "1 – Some difficulty" },
    { value: 2, label: "2 – A lot of difficulty" },
    { value: 3, label: "3 – Cannot do" },
    { value: 4, label: "4 – Don't do" },
    { value: 5, label: "5 – N/A" },
  ];

  const selfRatedOptions = [
    { value: 1, label: "1 – Excellent" },
    { value: 2, label: "2 – Very Good" },
    { value: 3, label: "3 – Good" },
    { value: 4, label: "4 – Fair" },
    { value: 5, label: "5 – Poor" },
  ];

  const maritalOptions = [
    { value: 1, label: "1 – Married" },
    { value: 2, label: "2 – Married, spouse absent" },
    { value: 3, label: "3 – Partnered" },
    { value: 4, label: "4 – Separated" },
    { value: 5, label: "5 – Divorced" },
    { value: 6, label: "6 – Widowed" },
    { value: 7, label: "7 – Never married" },
    { value: 8, label: "8 – Unknown" },
  ];

  const healthCompOptions = [
    { value: 1, label: "1 – Better" },
    { value: 2, label: "2 – About the same" },
    { value: 3, label: "3 – Worse" },
  ];

  const yesNoOptions = [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ];

  // -----------------------------------------------------------------------
  // RENDER
  // -----------------------------------------------------------------------

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      noValidate
      className="mx-auto max-w-2xl space-y-10 rounded-2xl bg-white p-4 shadow-xl sm:p-8"
    >
      {/* ---------- HEADER ---------- */}
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-green-dark sm:text-3xl">
          Health &amp; Wellbeing Assessment
        </h1>
        <p className="text-sm text-gray-500">
          Self-reported health intake form for adults 50+. All fields are
          required unless noted otherwise.
        </p>
      </div>

      {/* ============================================================== */}
      {/* SECTION: General Health                                         */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          General Health
        </legend>

        {/* Self-rated health */}
        <FieldWrapper
          id="SHLT"
          label="Self-rated health"
          helper="How would you rate your health? (1 Excellent – 5 Poor)"
          error={err("SHLT")}
        >
          <select
            id="SHLT"
            {...register("SHLT")}
            aria-describedby="SHLT-helper SHLT-error"
            aria-invalid={!!errors.SHLT}
            className={selectBase}
          >
            <option value="">Select…</option>
            {selfRatedOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </FieldWrapper>

        {/* Self-rated health compared */}
        <FieldWrapper
          id="SHLTC"
          label="Health compared to 2 years ago"
          helper="1 Better, 2 Same, 3 Worse"
          error={err("SHLTC")}
        >
          <select
            id="SHLTC"
            {...register("SHLTC")}
            aria-describedby="SHLTC-helper SHLTC-error"
            aria-invalid={!!errors.SHLTC}
            className={selectBase}
          >
            <option value="">Select…</option>
            {healthCompOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </FieldWrapper>

        {/* Condition count */}
        <FieldWrapper
          id="CONDE"
          label="Number of chronic conditions"
          helper="0–10"
          error={err("CONDE")}
        >
          <input
            id="CONDE"
            type="number"
            min={0}
            max={10}
            step={1}
            {...register("CONDE")}
            aria-describedby="CONDE-helper CONDE-error"
            aria-invalid={!!errors.CONDE}
            className={inputBase}
          />
        </FieldWrapper>
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Body Measures                                          */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Body Measures
        </legend>

        <div className="grid gap-4 sm:grid-cols-2">
          {/* Weight */}
          <FieldWrapper
            id="WEIGHT"
            label="Weight (kg)"
            helper="30–250 kg"
            error={err("WEIGHT")}
          >
            <input
              id="WEIGHT"
              type="number"
              min={30}
              max={250}
              step={0.1}
              {...register("WEIGHT")}
              aria-describedby="WEIGHT-helper WEIGHT-error"
              aria-invalid={!!errors.WEIGHT}
              className={inputBase}
            />
          </FieldWrapper>

          {/* Height */}
          <FieldWrapper
            id="HEIGHT"
            label="Height (cm)"
            helper="120–220 cm"
            error={err("HEIGHT")}
          >
            <input
              id="HEIGHT"
              type="number"
              min={120}
              max={220}
              step={0.1}
              {...register("HEIGHT")}
              aria-describedby="HEIGHT-helper HEIGHT-error"
              aria-invalid={!!errors.HEIGHT}
              className={inputBase}
            />
          </FieldWrapper>
        </div>

        {/* BMI */}
        <FieldWrapper
          id="BMI"
          label="BMI"
          helper={
            showComputedBMI
              ? `Auto-computed from weight & height: ${computedBMI}`
              : "10–60. Leave blank to auto-compute from weight & height."
          }
          error={err("BMI")}
        >
          {showComputedBMI ? (
            <input
              id="BMI"
              type="number"
              readOnly
              value={computedBMI ?? ""}
              aria-describedby="BMI-helper BMI-error"
              className={`${inputBase} bg-gray-100 font-semibold`}
              tabIndex={-1}
            />
          ) : (
            <input
              id="BMI"
              type="number"
              min={10}
              max={60}
              step={0.1}
              {...register("BMI")}
              aria-describedby="BMI-helper BMI-error"
              aria-invalid={!!errors.BMI}
              className={inputBase}
            />
          )}
        </FieldWrapper>
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Functional Limitations                                 */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Functional Limitations
        </legend>
        <p className="text-xs text-gray-500">
          0 = No difficulty, 1 = Some difficulty, 2 = A lot of difficulty,
          3 = Cannot do, 4 = Don&apos;t do, 5 = N/A
        </p>

        <div className="grid gap-4 sm:grid-cols-2">
          {(
            [
              ["MOBILA", "Mobility"],
              ["GROSSA", "Gross motor"],
              ["LGMUSA", "Large muscle"],
              ["FINEA", "Fine motor"],
              ["ADL5A", "ADL limitations (0–5)"],
              ["IADL5A", "IADL limitations (0–5)"],
            ] as const
          ).map(([key, label]) => (
            <FieldWrapper key={key} id={key} label={label} error={err(key)}>
              <select
                id={key}
                {...register(key)}
                aria-describedby={`${key}-error`}
                aria-invalid={!!errors[key]}
                className={selectBase}
              >
                {difficultyOptions.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </FieldWrapper>
          ))}
        </div>
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Cognition                                              */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Cognition
        </legend>

        <div className="grid gap-4 sm:grid-cols-2">
          {(
            [
              ["COG27", "Total cognition score (TICS)", 0, 27],
              ["TR20", "Total word recall (0–20)", 0, 20],
              ["IMRC", "Immediate recall (0–10)", 0, 10],
              ["DLRC", "Delayed recall (0–10)", 0, 10],
              ["SER7", "Serial 7s (0–5)", 0, 5],
            ] as const
          ).map(([key, label, min, max]) => (
            <FieldWrapper
              key={key}
              id={key}
              label={label}
              helper={`${min}–${max}`}
              error={err(key)}
            >
              <input
                id={key}
                type="number"
                min={min}
                max={max}
                step={1}
                {...register(key)}
                aria-describedby={`${key}-helper ${key}-error`}
                aria-invalid={!!errors[key]}
                className={inputBase}
              />
            </FieldWrapper>
          ))}
        </div>
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Depression (CES-D 8)                                   */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Depression (CES-D 8)
        </legend>

        <FieldWrapper
          id="CESD"
          label="CES-D score"
          helper="Sum of 8 binary items (0–8)"
          error={err("CESD")}
        >
          <input
            id="CESD"
            type="number"
            min={0}
            max={8}
            step={1}
            {...register("CESD")}
            aria-describedby="CESD-helper CESD-error"
            aria-invalid={!!errors.CESD}
            className={inputBase}
          />
        </FieldWrapper>

        <p className="text-xs text-gray-500">
          Individual CES-D items (0 = No, 1 = Yes):
        </p>

        <div className="grid gap-4 sm:grid-cols-2">
          {(
            [
              ["DEPRES", "Felt depressed"],
              ["EFFORT", "Everything felt like an effort"],
              ["SLEEPR", "Sleep was restless"],
              ["FLONE", "Felt lonely"],
            ] as const
          ).map(([key, label]) => (
            <FieldWrapper key={key} id={key} label={label} error={err(key)}>
              <select
                id={key}
                {...register(key)}
                aria-describedby={`${key}-error`}
                aria-invalid={!!errors[key]}
                className={selectBase}
              >
                {yesNoOptions.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </FieldWrapper>
          ))}
        </div>
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Smoking                                                */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Smoking
        </legend>

        {/* Ever smoked */}
        <Controller
          control={control}
          name="SMOKEV"
          render={({ field }) => (
            <FieldWrapper
              id="SMOKEV"
              label="Have you ever smoked?"
              error={err("SMOKEV")}
            >
              <div className="flex gap-6" role="radiogroup" aria-labelledby="SMOKEV">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="SMOKEV"
                    value="true"
                    checked={field.value === true}
                    onChange={() => field.onChange(true)}
                    className="accent-green-mid h-4 w-4"
                  />
                  Yes
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="SMOKEV"
                    value="false"
                    checked={field.value === false}
                    onChange={() => field.onChange(false)}
                    className="accent-green-mid h-4 w-4"
                  />
                  No
                </label>
              </div>
            </FieldWrapper>
          )}
        />

        {/* Current smoker – only visible if ever_smoked */}
        {everSmoked && (
          <Controller
            control={control}
            name="SMOKEN"
            render={({ field }) => (
              <FieldWrapper
                id="SMOKEN"
                label="Do you currently smoke?"
                error={err("SMOKEN")}
              >
                <div className="flex gap-6" role="radiogroup" aria-labelledby="SMOKEN">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="radio"
                      name="SMOKEN"
                      value="true"
                      checked={field.value === true}
                      onChange={() => field.onChange(true)}
                      className="accent-green-mid h-4 w-4"
                    />
                    Yes
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="radio"
                      name="SMOKEN"
                      value="false"
                      checked={field.value === false}
                      onChange={() => field.onChange(false)}
                      className="accent-green-mid h-4 w-4"
                    />
                    No
                  </label>
                </div>
              </FieldWrapper>
            )}
          />
        )}
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Alcohol                                                */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Alcohol Consumption
        </legend>

        <FieldWrapper
          id="DRINKN"
          label="Days per week you drink alcohol"
          helper="0–7"
          error={err("DRINKN")}
        >
          <input
            id="DRINKN"
            type="number"
            min={0}
            max={7}
            step={1}
            {...register("DRINKN")}
            aria-describedby="DRINKN-helper DRINKN-error"
            aria-invalid={!!errors.DRINKN}
            className={inputBase}
          />
        </FieldWrapper>

        {/* drinks_per_day – hidden when drink_days == 0 */}
        {drinkDays > 0 && (
          <FieldWrapper
            id="DRINKD"
            label="Drinks per day (on days you drink)"
            helper="0–20"
            error={err("DRINKD")}
          >
            <input
              id="DRINKD"
              type="number"
              min={0}
              max={20}
              step={1}
              {...register("DRINKD")}
              aria-describedby="DRINKD-helper DRINKD-error"
              aria-invalid={!!errors.DRINKD}
              className={inputBase}
            />
          </FieldWrapper>
        )}
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Physical Activity                                      */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Physical Activity
        </legend>

        <FieldWrapper
          id="VGACTX"
          label="Days per week of vigorous activity"
          helper="0–7"
          error={err("VGACTX")}
        >
          <input
            id="VGACTX"
            type="number"
            min={0}
            max={7}
            step={1}
            {...register("VGACTX")}
            aria-describedby="VGACTX-helper VGACTX-error"
            aria-invalid={!!errors.VGACTX}
            className={inputBase}
          />
        </FieldWrapper>
      </fieldset>

      {/* ============================================================== */}
      {/* SECTION: Sociodemographic & Financial                           */}
      {/* ============================================================== */}
      <fieldset className="space-y-4">
        <legend className="mb-2 text-lg font-bold text-green-dark border-b border-gray-200 pb-1 w-full">
          Sociodemographic &amp; Financial
        </legend>

        {/* Marital status */}
        <FieldWrapper
          id="MSTAT"
          label="Marital status"
          error={err("MSTAT")}
        >
          <select
            id="MSTAT"
            {...register("MSTAT")}
            aria-describedby="MSTAT-error"
            aria-invalid={!!errors.MSTAT}
            className={selectBase}
          >
            <option value="">Select…</option>
            {maritalOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </FieldWrapper>

        {/* Working */}
        <FieldWrapper
          id="WORK"
          label="Currently working for pay?"
          error={err("WORK")}
        >
          <select
            id="WORK"
            {...register("WORK")}
            aria-describedby="WORK-error"
            aria-invalid={!!errors.WORK}
            className={selectBase}
          >
            {yesNoOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </FieldWrapper>

        {/* Out of pocket */}
        <FieldWrapper
          id="OOPMD"
          label="Out-of-pocket medical expenses ($)"
          helper="Annual amount, USD ≥ 0"
          error={err("OOPMD")}
        >
          <div className="relative">
            <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-gray-400">
              $
            </span>
            <input
              id="OOPMD"
              type="number"
              min={0}
              step={0.01}
              {...register("OOPMD")}
              aria-describedby="OOPMD-helper OOPMD-error"
              aria-invalid={!!errors.OOPMD}
              className={`${inputBase} pl-7`}
            />
          </div>
        </FieldWrapper>
      </fieldset>

      {/* ============================================================== */}
      {/* SUBMIT                                                          */}
      {/* ============================================================== */}
      <button
        type="submit"
        className="w-full rounded-xl bg-green-dark px-6 py-3 text-base font-bold text-white shadow-md transition hover:bg-green-mid focus:outline-none focus:ring-2 focus:ring-green-mid focus:ring-offset-2 active:scale-[0.98] sm:w-auto"
      >
        Submit Assessment
      </button>
    </form>
  );
}
