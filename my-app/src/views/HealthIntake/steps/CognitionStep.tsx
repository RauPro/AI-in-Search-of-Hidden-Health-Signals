import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import {
  FieldSet,
  FieldLegend,
  FieldDescription,
} from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";

const FIELDS: {
  key: keyof HealthFormValues;
  label: string;
  helper: string;
  min: number;
  max: number;
}[] = [
  {
    key: "COG27",
    label: "Overall thinking score",
    helper: "From the TICS screening (0–27)",
    min: 0,
    max: 27,
  },
  {
    key: "TR20",
    label: "Word memory — total",
    helper: "Total words remembered across all rounds (0–20)",
    min: 0,
    max: 20,
  },
  {
    key: "IMRC",
    label: "Words recalled right away",
    helper: "Words remembered right after hearing them (0–10)",
    min: 0,
    max: 10,
  },
  {
    key: "DLRC",
    label: "Words recalled later",
    helper: "Words remembered after a short delay (0–10)",
    min: 0,
    max: 10,
  },
  {
    key: "SER7",
    label: "Counting backwards",
    helper: "Subtracting by 7 starting from 100 (0–5)",
    min: 0,
    max: 5,
  },
];

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
        Memory &amp; Thinking
      </FieldLegend>
      <FieldDescription>
        If you&apos;ve completed a cognitive screening, enter your scores below.
        Your healthcare provider can help if you&apos;re unsure.
      </FieldDescription>

      <div className="grid gap-4 sm:grid-cols-2">
        {FIELDS.map(({ key, label, helper, min, max }) => (
          <FieldWrapper
            key={key}
            id={key}
            label={label}
            helper={helper}
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
