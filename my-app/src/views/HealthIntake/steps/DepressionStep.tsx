import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { NativeSelect } from "@/components/ui/native-select";
import {
  FieldSet,
  FieldLegend,
  FieldDescription,
} from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";
import { yesNoOptions } from "../utils";

const FEELING_ITEMS: {
  key: keyof HealthFormValues;
  label: string;
  helper: string;
}[] = [
  {
    key: "DEPRES",
    label: "Felt sad or down",
    helper: "Did you feel depressed much of the time?",
  },
  {
    key: "EFFORT",
    label: "Everything felt like a struggle",
    helper: "Did everyday tasks feel harder than usual?",
  },
  {
    key: "SLEEPR",
    label: "Had trouble sleeping",
    helper: "Was your sleep restless or disturbed?",
  },
  {
    key: "FLONE",
    label: "Felt alone or isolated",
    helper: "Did you feel lonely much of the time?",
  },
];

export default function DepressionStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        How You&apos;ve Been Feeling
      </FieldLegend>
      <FieldDescription>
        Think about the past week. There are no right or wrong answers — just
        answer honestly.
      </FieldDescription>

      <FieldWrapper
        id="CESD"
        label="Overall mood score"
        helper="How many of these feelings have you experienced in the past week? (0–8)"
        error={err("CESD")}
      >
        <Input
          id="CESD"
          type="number"
          min={0}
          max={8}
          step={1}
          {...register("CESD")}
          aria-describedby="CESD-helper CESD-error"
          aria-invalid={!!errors.CESD || undefined}
        />
      </FieldWrapper>

      <FieldDescription>
        Have any of these been true for you recently?
      </FieldDescription>

      <div className="grid gap-4 sm:grid-cols-2">
        {FEELING_ITEMS.map(({ key, label, helper }) => (
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
              {yesNoOptions.map((o) => (
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
