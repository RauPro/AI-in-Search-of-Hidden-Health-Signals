import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { NativeSelect } from "@/components/ui/native-select";
import { FieldSet, FieldLegend } from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";
import { selfRatedOptions, healthCompOptions } from "../utils";

export default function GeneralHealthStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        General Health
      </FieldLegend>

      {/* Self-rated health */}
      <FieldWrapper
        id="SHLT"
        label="Self-rated health"
        helper="How would you rate your health? (1 Excellent – 5 Poor)"
        error={err("SHLT")}
      >
        <NativeSelect
          id="SHLT"
          {...register("SHLT")}
          aria-describedby="SHLT-helper SHLT-error"
          aria-invalid={!!errors.SHLT || undefined}
        >
          <option value="">Select&hellip;</option>
          {selfRatedOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </NativeSelect>
      </FieldWrapper>

      {/* Self-rated health compared */}
      <FieldWrapper
        id="SHLTC"
        label="Health compared to 2 years ago"
        error={err("SHLTC")}
      >
        <NativeSelect
          id="SHLTC"
          {...register("SHLTC")}
          aria-describedby="SHLTC-error"
          aria-invalid={!!errors.SHLTC || undefined}
        >
          <option value="">Select&hellip;</option>
          {healthCompOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </NativeSelect>
      </FieldWrapper>

      {/* Condition count */}
      <FieldWrapper
        id="CONDE"
        label="Number of chronic conditions"
        error={err("CONDE")}
      >
        <Input
          id="CONDE"
          type="number"
          min={0}
          step={1}
          {...register("CONDE")}
          aria-describedby="CONDE-error"
          aria-invalid={!!errors.CONDE || undefined}
        />
      </FieldWrapper>
    </FieldSet>
  );
}
