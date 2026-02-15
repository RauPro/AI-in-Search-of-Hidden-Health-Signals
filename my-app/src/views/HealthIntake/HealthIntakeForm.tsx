"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { MultiStepForm } from "@/components/layout";
import { healthSchema, type HealthFormValues, STEP_FIELDS } from "./schema";
import { computeBMI, computeAge } from "./utils";
import {
  DemographicsStep,
  GeneralHealthStep,
  BodyMeasuresStep,
  FunctionalLimitationsStep,
  CognitionStep,
  DepressionStep,
  LifestyleStep,
  SociodemographicStep,
} from "./steps";

// ---------------------------------------------------------------------------
// Step configuration
// ---------------------------------------------------------------------------
const STEPS = [
  {
    id: "demographics",
    title: "Demographics",
    fields: STEP_FIELDS.demographics,
  },
  {
    id: "general",
    title: "General Health",
    fields: STEP_FIELDS.generalHealth,
  },
  {
    id: "body",
    title: "Body Measures",
    fields: STEP_FIELDS.bodyMeasures,
  },
  {
    id: "functional",
    title: "Daily Activities",
    fields: STEP_FIELDS.functionalLimitations,
  },
  {
    id: "cognition",
    title: "Memory & Thinking",
    fields: STEP_FIELDS.cognition,
  },
  {
    id: "depression",
    title: "Depression",
    fields: STEP_FIELDS.depression,
  },
  {
    id: "lifestyle",
    title: "Lifestyle",
    fields: STEP_FIELDS.lifestyle,
  },
  {
    id: "socio",
    title: "Sociodemographic & Financial",
    fields: STEP_FIELDS.sociodemographic,
  },
];

// ---------------------------------------------------------------------------
// Stub submit
// ---------------------------------------------------------------------------
type SubmitPayload = HealthFormValues & {
  BMI: number | null;
  age: number | null;
  age_squared: number | null;
  female: number;
};

function submitAssessment(payload: SubmitPayload) {
  // TODO: wire to API
  console.log("submitAssessment payload:", payload);
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function HealthIntakeForm() {
  const [currentStep, setCurrentStep] = useState(0);

  const form = useForm<HealthFormValues>({
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    resolver: zodResolver(healthSchema) as any,
    defaultValues: {
      BIRTH_YEAR: undefined,
      RAGENDER: undefined,
      ETHNICITY: undefined,
      RAEDYRS: 0,
      RAEDUC: undefined,
      RAEDEGRM: 0,
      SHLT: undefined,
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
    mode: "onTouched",
  });

  const { handleSubmit, trigger, watch } = form;

  // Auto-compute age from birth year
  const birthYear = watch("BIRTH_YEAR");
  const computedAge =
    birthYear && birthYear >= 1920 && birthYear <= 2010
      ? computeAge(birthYear)
      : null;

  // Auto-compute BMI from weight & height
  const weight = watch("WEIGHT");
  const height = watch("HEIGHT");
  const computedBMI =
    weight && height && weight >= 30 && height >= 120
      ? computeBMI(weight, height)
      : null;

  // -- Step navigation -------------------------------------------------------
  const handleNext = async () => {
    const stepFields = STEPS[currentStep].fields;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const isValid = await trigger(stepFields as any);
    if (isValid) setCurrentStep((s) => Math.min(s + 1, STEPS.length - 1));
  };

  const handleBack = () => {
    setCurrentStep((s) => Math.max(s - 1, 0));
  };

  // -- Submit ----------------------------------------------------------------
  const onSubmit = handleSubmit((data) => {
    const payload = {
      ...data,
      BMI: computedBMI,
      age: computedAge,
      age_squared: computedAge !== null ? computedAge ** 2 : null,
      female: data.RAGENDER === 2 ? 1 : 0,
    };
    submitAssessment(payload);
  });

  // -- Render current step ---------------------------------------------------
  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return <DemographicsStep form={form} />;
      case 1:
        return <GeneralHealthStep form={form} />;
      case 2:
        return <BodyMeasuresStep form={form} />;
      case 3:
        return <FunctionalLimitationsStep form={form} />;
      case 4:
        return <CognitionStep form={form} />;
      case 5:
        return <DepressionStep form={form} />;
      case 6:
        return <LifestyleStep form={form} />;
      case 7:
        return <SociodemographicStep form={form} />;
      default:
        return null;
    }
  };

  return (
    <MultiStepForm
      steps={STEPS}
      currentStep={currentStep}
      onNext={handleNext}
      onBack={handleBack}
      onSubmit={onSubmit}
    >
      {/* Header */}
      <div className="mb-6 space-y-1">
        <h1 className="text-2xl font-bold text-primary sm:text-3xl">
          Health &amp; Wellbeing Assessment
        </h1>
        <p className="text-muted-foreground text-sm">
          Self-reported health intake form for adults 50+. All fields are
          required unless noted otherwise.
        </p>
      </div>

      {renderStep()}
    </MultiStepForm>
  );
}
