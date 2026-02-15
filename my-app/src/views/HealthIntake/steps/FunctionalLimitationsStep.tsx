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

const FIELDS: {
  key: keyof HealthFormValues;
  label: string;
  helper: string;
}[] = [
  {
    key: "MOBILA",
    label: "Getting around",
    helper: "e.g. walking across a room, going outside",
  },
  {
    key: "GROSSA",
    label: "Large movements",
    helper: "e.g. climbing stairs, getting out of a chair",
  },
  {
    key: "LGMUSA",
    label: "Lifting & carrying",
    helper: "e.g. carrying groceries, pushing a vacuum",
  },
  {
    key: "FINEA",
    label: "Using your hands",
    helper: "e.g. buttoning a shirt, picking up a coin",
  },
  {
    key: "ADL5A",
    label: "Basic self-care",
    helper: "e.g. bathing, dressing, eating",
  },
  {
    key: "IADL5A",
    label: "Household tasks",
    helper: "e.g. cooking, managing money, using the phone",
  },
];

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
        Your Everyday Abilities
      </FieldLegend>
      <FieldDescription>
        Think about your typical day. How easy or hard are these activities for
        you right now?
      </FieldDescription>

      <div className="grid gap-4 sm:grid-cols-2">
        {FIELDS.map(({ key, label, helper }) => (
          <FieldWrapper
            key={key}
            id={key}
            label={label}
            helper={helper}
            error={err(key)}
          >
            <NativeSelect
              id={key}
              {...register(key)}
              aria-describedby={`${key}-helper ${key}-error`}
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
