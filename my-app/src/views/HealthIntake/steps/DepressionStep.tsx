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

const CESD_ITEMS = [
  ["DEPRES", "Felt depressed"],
  ["EFFORT", "Everything felt like an effort"],
  ["SLEEPR", "Sleep was restless"],
  ["FLONE", "Felt lonely"],
] as const;

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
        Depression (CES-D 8)
      </FieldLegend>

      <FieldWrapper
        id="CESD"
        label="CES-D score"
        helper="Sum of 8 binary items (0–8)"
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
        Individual CES-D items (0 = No, 1 = Yes):
      </FieldDescription>

      <div className="grid gap-4 sm:grid-cols-2">
        {CESD_ITEMS.map(([key, label]) => (
          <FieldWrapper key={key} id={key} label={label} error={err(key)}>
            <NativeSelect
              id={key}
              {...register(key)}
              aria-describedby={`${key}-error`}
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
