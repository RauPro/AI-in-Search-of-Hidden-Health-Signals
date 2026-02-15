import { FieldWrapper } from "@/components/forms";
import { NativeSelect } from "@/components/ui/native-select";
import {
  FieldSet,
  FieldLegend,
  FieldDescription,
} from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";
import { difficultyOptions } from "../utils";

const FIELDS = [
  ["MOBILA", "Mobility"],
  ["GROSSA", "Gross motor"],
  ["LGMUSA", "Large muscle"],
  ["FINEA", "Fine motor"],
  ["ADL5A", "ADL limitations (0–5)"],
  ["IADL5A", "IADL limitations (0–5)"],
] as const;

export default function FunctionalLimitationsStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        Functional Limitations
      </FieldLegend>
      <FieldDescription>
        0 = No difficulty, 1 = Some difficulty, 2 = A lot of difficulty, 3 =
        Cannot do, 4 = Don&apos;t do, 5 = N/A
      </FieldDescription>

      <div className="grid gap-4 sm:grid-cols-2">
        {FIELDS.map(([key, label]) => (
          <FieldWrapper key={key} id={key} label={label} error={err(key)}>
            <NativeSelect
              id={key}
              {...register(key)}
              aria-describedby={`${key}-error`}
              aria-invalid={!!errors[key] || undefined}
            >
              {difficultyOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </NativeSelect>
          </FieldWrapper>
        ))}
      </div>
    </FieldSet>
  );
}
