import React from "react";
import { Progress } from "@/components/ui/progress";

export type StepConfig = {
  id: string;
  title: string;
};

type StepIndicatorProps = {
  steps: StepConfig[];
  currentStep: number;
};

export default function StepIndicator({
  steps,
  currentStep,
}: StepIndicatorProps) {
  const progressValue = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="space-y-3">
      <Progress value={progressValue} className="h-2" />
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground text-xs font-medium">
          Step {currentStep + 1} of {steps.length}
        </span>
        <span className="text-primary text-xs font-semibold">
          {steps[currentStep].title}
        </span>
      </div>
    </div>
  );
}
