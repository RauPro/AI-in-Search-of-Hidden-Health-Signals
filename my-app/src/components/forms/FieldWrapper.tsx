import React from "react";
import {
  Field,
  FieldLabel,
  FieldDescription,
  FieldError,
} from "@/components/ui/field";

type FieldWrapperProps = {
  id: string;
  label: string;
  helper?: string;
  error?: string;
  children: React.ReactNode;
};

export default function FieldWrapper({
  id,
  label,
  helper,
  error,
  children,
}: FieldWrapperProps) {
  return (
    <Field data-invalid={!!error || undefined}>
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      {children}
      {helper && (
        <FieldDescription id={`${id}-helper`}>{helper}</FieldDescription>
      )}
      {error && <FieldError id={`${id}-error`}>{error}</FieldError>}
    </Field>
  );
}
