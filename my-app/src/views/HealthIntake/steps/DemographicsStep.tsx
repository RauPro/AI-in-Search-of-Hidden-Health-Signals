import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { NativeSelect } from "@/components/ui/native-select";
import { FieldSet, FieldLegend } from "@/components/ui/field";
import type { HealthFormValues } from "../schema";
import type { StepProps } from "../types";
import {
  genderOptions,
  ethnicityOptions,
  educationCategoryOptions,
  degreeOptions,
} from "../utils";

export default function DemographicsStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        Demographics
      </FieldLegend>

      <div className="grid gap-4">
        {/* Birth year */}
        <FieldWrapper
          id="BIRTH_YEAR"
          label="Birth year"
          error={err("BIRTH_YEAR")}
        >
          <Input
            id="BIRTH_YEAR"
            type="number"
            min={1900}
            step={1}
            {...register("BIRTH_YEAR")}
            aria-describedby="BIRTH_YEAR-error"
            aria-invalid={!!errors.BIRTH_YEAR || undefined}
          />
        </FieldWrapper>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* Gender */}
        <FieldWrapper id="RAGENDER" label="Gender" error={err("RAGENDER")}>
          <NativeSelect
            id="RAGENDER"
            {...register("RAGENDER")}
            aria-describedby="RAGENDER-error"
            aria-invalid={!!errors.RAGENDER || undefined}
          >
            <option value="">Select&hellip;</option>
            {genderOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </NativeSelect>
        </FieldWrapper>

        {/* Ethnicity */}
        <FieldWrapper id="ETHNICITY" label="Ethnicity" error={err("ETHNICITY")}>
          <NativeSelect
            id="ETHNICITY"
            {...register("ETHNICITY")}
            aria-describedby="ETHNICITY-error"
            aria-invalid={!!errors.ETHNICITY || undefined}
          >
            <option value="">Select&hellip;</option>
            {ethnicityOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </NativeSelect>
        </FieldWrapper>
      </div>

      {/* Years of education */}
      <FieldWrapper
        id="RAEDYRS"
        label="Years of education"
        error={err("RAEDYRS")}
      >
        <Input
          id="RAEDYRS"
          type="number"
          min={0}
          step={1}
          {...register("RAEDYRS")}
          aria-describedby="RAEDYRS-error"
          aria-invalid={!!errors.RAEDYRS || undefined}
        />
      </FieldWrapper>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* Education category */}
        <FieldWrapper
          id="RAEDUC"
          label="Education category"
          error={err("RAEDUC")}
        >
          <NativeSelect
            id="RAEDUC"
            {...register("RAEDUC")}
            aria-describedby="RAEDUC-error"
            aria-invalid={!!errors.RAEDUC || undefined}
          >
            <option value="">Select&hellip;</option>
            {educationCategoryOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </NativeSelect>
        </FieldWrapper>

        {/* Highest degree */}
        <FieldWrapper
          id="RAEDEGRM"
          label="Highest degree"
          error={err("RAEDEGRM")}
        >
          <NativeSelect
            id="RAEDEGRM"
            {...register("RAEDEGRM")}
            aria-describedby="RAEDEGRM-error"
            aria-invalid={!!errors.RAEDEGRM || undefined}
          >
            {degreeOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </NativeSelect>
        </FieldWrapper>
      </div>
    </FieldSet>
  );
}
