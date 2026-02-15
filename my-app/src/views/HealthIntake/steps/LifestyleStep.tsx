import { useEffect } from "react";
import { Controller } from "react-hook-form";
import { FieldWrapper } from "@/components/forms";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import {
  Field,
  FieldError,
  FieldLabel,
  FieldSet,
  FieldLegend,
} from "@/components/ui/field";
import type { StepProps } from "../types";
import type { HealthFormValues } from "../schema";

export default function LifestyleStep({ form }: StepProps) {
  const {
    register,
    control,
    watch,
    setValue,
    formState: { errors },
  } = form;

  const err = (field: keyof HealthFormValues) =>
    errors[field]?.message as string | undefined;

  const everSmoked = watch("SMOKEV");
  const drinkDays = watch("DRINKN");

  // Reset current_smoker when ever_smoked is toggled off
  useEffect(() => {
    if (!everSmoked) setValue("SMOKEN", false);
  }, [everSmoked, setValue]);

  // Force drinks_per_day to 0 when drink_days_week == 0
  useEffect(() => {
    if (drinkDays === 0) setValue("DRINKD", 0);
  }, [drinkDays, setValue]);

  return (
    <div className="space-y-8">
      {/* ---- Smoking ---- */}
      <FieldSet className="gap-4">
        <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
          Smoking
        </FieldLegend>

        {/* Ever smoked */}
        <Controller
          control={control}
          name="SMOKEV"
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid || undefined}>
              <FieldLabel>Have you ever smoked?</FieldLabel>
              <RadioGroup
                value={field.value ? "true" : "false"}
                onValueChange={(val) => field.onChange(val === "true")}
                className="flex flex-row gap-4"
              >
                <div className="flex items-center gap-2">
                  <RadioGroupItem
                    value="true"
                    id="SMOKEV-yes"
                    aria-invalid={fieldState.invalid || undefined}
                  />
                  <Label htmlFor="SMOKEV-yes">Yes</Label>
                </div>
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="false" id="SMOKEV-no" />
                  <Label htmlFor="SMOKEV-no">No</Label>
                </div>
              </RadioGroup>
              {fieldState.invalid && (
                <FieldError>{fieldState.error?.message}</FieldError>
              )}
            </Field>
          )}
        />

        {/* Current smoker – only visible if ever_smoked */}
        {everSmoked && (
          <Controller
            control={control}
            name="SMOKEN"
            render={({ field, fieldState }) => (
              <Field data-invalid={fieldState.invalid || undefined}>
                <FieldLabel>Do you currently smoke?</FieldLabel>
                <RadioGroup
                  value={field.value ? "true" : "false"}
                  onValueChange={(val) => field.onChange(val === "true")}
                  className="flex flex-row gap-4"
                >
                  <div className="flex items-center gap-2">
                    <RadioGroupItem
                      value="true"
                      id="SMOKEN-yes"
                      aria-invalid={fieldState.invalid || undefined}
                    />
                    <Label htmlFor="SMOKEN-yes">Yes</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <RadioGroupItem value="false" id="SMOKEN-no" />
                    <Label htmlFor="SMOKEN-no">No</Label>
                  </div>
                </RadioGroup>
                {fieldState.invalid && (
                  <FieldError>{fieldState.error?.message}</FieldError>
                )}
              </Field>
            )}
          />
        )}
      </FieldSet>

      {/* ---- Alcohol ---- */}
      <FieldSet className="gap-4">
        <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
          Alcohol Consumption
        </FieldLegend>

        <FieldWrapper
          id="DRINKN"
          label="Days per week you drink alcohol"
          error={err("DRINKN")}
        >
          <Input
            id="DRINKN"
            type="number"
            min={0}
            max={7}
            step={1}
            {...register("DRINKN")}
            aria-describedby="DRINKN-helper DRINKN-error"
            aria-invalid={!!errors.DRINKN || undefined}
          />
        </FieldWrapper>

        {/* drinks_per_day – hidden when drink_days == 0 */}
        {drinkDays > 0 && (
          <FieldWrapper
            id="DRINKD"
            label="Drinks per day (on days you drink)"
            helper="0–20"
            error={err("DRINKD")}
          >
            <Input
              id="DRINKD"
              type="number"
              min={0}
              max={20}
              step={1}
              {...register("DRINKD")}
              aria-describedby="DRINKD-helper DRINKD-error"
              aria-invalid={!!errors.DRINKD || undefined}
            />
          </FieldWrapper>
        )}
      </FieldSet>

      {/* ---- Physical Activity ---- */}
      <FieldSet className="gap-4">
        <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
          Physical Activity
        </FieldLegend>

        <FieldWrapper
          id="VGACTX"
          label="Days per week of vigorous activity"
          error={err("VGACTX")}
        >
          <Input
            id="VGACTX"
            type="number"
            min={0}
            max={7}
            step={1}
            {...register("VGACTX")}
            aria-describedby="VGACTX-helper VGACTX-error"
            aria-invalid={!!errors.VGACTX || undefined}
          />
        </FieldWrapper>
      </FieldSet>
    </div>
  );
}
