import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { FieldSet, FieldLegend } from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";

export default function BodyMeasuresStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        Body Measures
      </FieldLegend>

      <div className="grid gap-4 sm:grid-cols-2">
        {/* Weight */}
        <FieldWrapper
          id="WEIGHT"
          label="Weight (kg)"
          helper="20–300 kg"
          error={err("WEIGHT")}
        >
          <Input
            id="WEIGHT"
            type="number"
            min={20}
            max={300}
            step={0.1}
            {...register("WEIGHT")}
            aria-describedby="WEIGHT-helper WEIGHT-error"
            aria-invalid={!!errors.WEIGHT || undefined}
          />
        </FieldWrapper>

        {/* Height */}
        <FieldWrapper
          id="HEIGHT"
          label="Height (cm)"
          helper="100–250 cm"
          error={err("HEIGHT")}
        >
          <Input
            id="HEIGHT"
            type="number"
            min={100}
            max={250}
            step={0.1}
            {...register("HEIGHT")}
            aria-describedby="HEIGHT-helper HEIGHT-error"
            aria-invalid={!!errors.HEIGHT || undefined}
          />
        </FieldWrapper>
      </div>

    </FieldSet>
  );
}
