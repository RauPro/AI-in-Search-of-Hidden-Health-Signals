import type { UseFormReturn } from "react-hook-form";
import type { HealthFormValues } from "./schema";

export type StepProps = {
  form: UseFormReturn<HealthFormValues>;
};
