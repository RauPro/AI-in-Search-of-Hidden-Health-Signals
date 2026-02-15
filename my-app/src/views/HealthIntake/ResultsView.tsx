import {
  Card,
  CardHeader,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PredictionResult = {
  status: string;
  health_score: number;
  risks: Record<string, number>;
  message: string;
};

type ResultsViewProps = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result: Record<string, any>;
  onStartOver: () => void;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RISK_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  cvd: "Cardiovascular Disease",
  stroke: "Stroke",
  lung: "Lung Disease",
  cancer: "Cancer",
  hibp: "High Blood Pressure",
  arthritis: "Arthritis",
  memory: "Memory Problems",
  psychiatric: "Psychiatric",
};

function getBarColor(value: number): string {
  if (value >= 30)
    return "[&>[data-slot=progress-indicator]]:bg-red-500";
  if (value >= 20)
    return "[&>[data-slot=progress-indicator]]:bg-amber-500";
  return "[&>[data-slot=progress-indicator]]:bg-green-600";
}

function getScoreColor(score: number): string {
  if (score >= 80) return "text-green-700";
  if (score >= 60) return "text-amber-600";
  return "text-red-700";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ResultsView({ result, onStartOver }: ResultsViewProps) {
  const data = result as PredictionResult;
  const score = data.health_score ?? 0;
  const risks = data.risks ?? {};

  // Sort risks from highest to lowest
  const sortedRisks = Object.entries(risks).sort(([, a], [, b]) => b - a);

  return (
    <Card className="mx-auto max-w-2xl">
      <CardHeader className="text-center">
        <h1 className="text-2xl font-bold text-primary sm:text-3xl">
          Your Health Assessment Results
        </h1>
        <p className="text-muted-foreground text-sm">
          Based on the information you provided, here is your health risk
          profile.
        </p>
      </CardHeader>

      <CardContent className="space-y-8">
        {/* Health Score */}
        <div className="flex flex-col items-center gap-3 rounded-xl border bg-muted/30 p-6">
          <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
            Overall Health Score
          </span>
          <span className={`text-6xl font-extrabold ${getScoreColor(score)}`}>
            {score.toFixed(1)}
          </span>
          <Progress value={score} className="h-3 w-full max-w-xs" />
          <span className="text-xs text-muted-foreground">out of 100</span>
        </div>

        <Separator />

        {/* Risk Predictions */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-primary">
            Risk Predictions
          </h2>

          <div className="space-y-3">
            {sortedRisks.map(([key, value]) => (
              <div key={key} className="space-y-1.5">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-medium">
                    {RISK_LABELS[key] ?? key}
                  </span>
                  <span className="text-sm font-semibold tabular-nums">
                    {value.toFixed(1)}%
                  </span>
                </div>
                <Progress
                  value={value}
                  className={`h-2 ${getBarColor(value)}`}
                />
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Disclaimer */}
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <p className="text-sm text-amber-800">
            {data.message ||
              "Remember: this is a screening tool, not a diagnosis. Please consult a healthcare professional for medical advice."}
          </p>
        </div>
      </CardContent>

      <CardFooter className="flex justify-center border-t pt-6">
        <Button onClick={onStartOver} size="lg">
          Start New Assessment
        </Button>
      </CardFooter>
    </Card>
  );
}
