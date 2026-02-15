import React from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import StepIndicator from "./StepIndicator";
import type { StepConfig } from "./StepIndicator";

type MultiStepFormProps = {
  steps: StepConfig[];
  currentStep: number;
  children: React.ReactNode;
  onNext: () => void;
  onBack: () => void;
  onSubmit: () => void;
  submitting?: boolean;
};

export default function MultiStepForm({
  steps,
  currentStep,
  children,
  onNext,
  onBack,
  onSubmit,
  submitting = false,
}: MultiStepFormProps) {
  const isFirst = currentStep === 0;
  const isLast = currentStep === steps.length - 1;

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isLast) {
      onSubmit();
    } else {
      onNext();
    }
  };

  return (
    <Card className="mx-auto max-w-2xl">
      <form onSubmit={handleFormSubmit} noValidate className="contents">
        <CardHeader>
          <StepIndicator steps={steps} currentStep={currentStep} />
        </CardHeader>

        <CardContent>{children}</CardContent>

        <CardFooter className="flex justify-between border-t pt-6">
          {isFirst ? (
            <div />
          ) : (
            <Button
              type="button"
              variant="outline"
              onClick={onBack}
              disabled={submitting}
            >
              Back
            </Button>
          )}

          {isLast ? (
            <Button type="submit" disabled={submitting}>
              {submitting ? "Submitting..." : "Submit"}
            </Button>
          ) : (
            <Button type="button" onClick={onNext}>
              Next
            </Button>
          )}
        </CardFooter>
      </form>
    </Card>
  );
}
