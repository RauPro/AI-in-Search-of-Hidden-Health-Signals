import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { FieldSet, FieldLegend } from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";

const FIELDS = [
  ["COG27", "Total cognition score (TICS)", 0, 27],
  ["TR20", "Total word recall (0–20)", 0, 20],
  ["IMRC", "Immediate recall (0–10)", 0, 10],
  ["DLRC", "Delayed recall (0–10)", 0, 10],
  ["SER7", "Serial 7s (0–5)", 0, 5],
] as const;

export default function CognitionStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        Cognition
      </FieldLegend>

      <div className="grid gap-4 sm:grid-cols-2">
        {FIELDS.map(([key, label, min, max]) => (
          <FieldWrapper
            key={key}
            id={key}
            label={label}
            helper={`${min}–${max}`}
            error={err(key)}
          >
            <Input
              id={key}
              type="number"
              min={min}
              max={max}
              step={1}
              {...register(key)}
              aria-describedby={`${key}-helper ${key}-error`}
              aria-invalid={!!errors[key] || undefined}
            />
          </FieldWrapper>
        ))}
      </div>
    </FieldSet>
  );
}
