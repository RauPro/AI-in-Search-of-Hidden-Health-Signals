import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { NativeSelect } from "@/components/ui/native-select";
import { FieldSet, FieldLegend } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";
import { maritalOptions, yesNoOptions } from "../utils";

export default function SociodemographicStep({ form }: StepProps) {
  const {
    register,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        Sociodemographic &amp; Financial
      </FieldLegend>

      {/* Marital status */}
      <FieldWrapper id="MSTAT" label="Marital status" error={err("MSTAT")}>
        <NativeSelect
          id="MSTAT"
          {...register("MSTAT")}
          aria-describedby="MSTAT-error"
          aria-invalid={!!errors.MSTAT || undefined}
        >
          <option value="">Select&hellip;</option>
          {maritalOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </NativeSelect>
      </FieldWrapper>

      {/* Working */}
      <FieldWrapper
        id="WORK"
        label="Currently working for pay?"
        error={err("WORK")}
      >
        <NativeSelect
          id="WORK"
          {...register("WORK")}
          aria-describedby="WORK-error"
          aria-invalid={!!errors.WORK || undefined}
        >
          {yesNoOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </NativeSelect>
      </FieldWrapper>

      {/* Out of pocket */}
      <FieldWrapper
        id="OOPMD"
        label="Out-of-pocket medical expenses ($)"
        helper="Annual amount, USD ≥ 0"
        error={err("OOPMD")}
      >
        <div className="relative">
          <Label className="text-muted-foreground pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm">
            $
          </Label>
          <Input
            id="OOPMD"
            type="number"
            min={0}
            step={0.01}
            {...register("OOPMD")}
            aria-describedby="OOPMD-helper OOPMD-error"
            aria-invalid={!!errors.OOPMD || undefined}
            className="pl-7"
          />
        </div>
      </FieldWrapper>
    </FieldSet>
  );
}
