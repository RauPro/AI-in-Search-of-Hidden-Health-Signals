"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { FieldSet, FieldLegend, FieldDescription } from "@/components/ui/field";
import type { StepProps } from "../types";
import { WORD_POOLS } from "../utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Phase =
  | "memorize"
  | "immediateRecall"
  | "serial7"
  | "countdown"
  | "delayedRecall"
  | "results";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Pick a random word pool (stable per mount via useMemo). */
function pickWords(): string[] {
  const idx = Math.floor(Math.random() * WORD_POOLS.length);
  return [...WORD_POOLS[idx]];
}

/** Shuffle an array (Fisher-Yates). */
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------------------------------------------------------------------------
// Sub-components for each phase
// ---------------------------------------------------------------------------

/** Phase 1 – Memorize words with a countdown timer. */
function MemorizePhase({
  words,
  onDone,
}: {
  words: string[];
  onDone: () => void;
}) {
  const DURATION = 15; // seconds
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setElapsed((e) => {
        if (e + 1 >= DURATION) {
          clearInterval(id);
          return DURATION;
        }
        return e + 1;
      });
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Auto-advance when timer completes
  const doneRef = useRef(false);
  useEffect(() => {
    if (elapsed >= DURATION && !doneRef.current) {
      doneRef.current = true;
      onDone();
    }
  }, [elapsed, onDone]);

  const pct = Math.round(((DURATION - elapsed) / DURATION) * 100);

  return (
    <div className="space-y-6">
      <div className="text-center space-y-1">
        <h3 className="text-lg font-semibold">Remember These Words</h3>
        <p className="text-sm text-muted-foreground">
          Study the words below. You&apos;ll be asked to recall them later.
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-3">
        {words.map((w, i) => (
          <span
            key={w}
            className="rounded-md border bg-muted px-4 py-2 text-base font-medium animate-in fade-in slide-in-from-bottom-2"
            style={{ animationDelay: `${i * 120}ms`, animationFillMode: "both" }}
          >
            {w}
          </span>
        ))}
      </div>

      <div className="space-y-1">
        <Progress value={pct} className="h-2" />
        <p className="text-xs text-center text-muted-foreground">
          {DURATION - elapsed}s remaining
        </p>
      </div>

      <div className="flex justify-center">
        <Button type="button" variant="outline" size="sm" onClick={onDone}>
          I&apos;m ready
        </Button>
      </div>
    </div>
  );
}

/** Phase 2 & 5 – Word recall input. */
function RecallPhase({
  title,
  description,
  targetWords,
  onDone,
}: {
  title: string;
  description: string;
  targetWords: string[];
  onDone: (score: number) => void;
}) {
  const [input, setInput] = useState("");
  const [guesses, setGuesses] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const targetSet = useMemo(
    () => new Set(targetWords.map((w) => w.toLowerCase())),
    [targetWords],
  );

  const submitWord = () => {
    const word = input.trim().toLowerCase();
    if (!word) return;
    if (!guesses.includes(word)) {
      setGuesses((prev) => [...prev, word]);
    }
    setInput("");
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submitWord();
    }
  };

  const correctCount = guesses.filter((g) => targetSet.has(g)).length;

  return (
    <div className="space-y-6">
      <div className="text-center space-y-1">
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>

      <div className="flex gap-2">
        <Input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a word and press Enter…"
          autoFocus
          className="flex-1"
        />
        <Button type="button" size="sm" onClick={submitWord}>
          Add
        </Button>
      </div>

      {guesses.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {guesses.map((g) => {
            const isCorrect = targetSet.has(g);
            return (
              <span
                key={g}
                className={`rounded-full px-3 py-1 text-sm font-medium ${
                  isCorrect
                    ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                    : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                }`}
              >
                {g}
              </span>
            );
          })}
        </div>
      )}

      <p className="text-sm text-muted-foreground text-center">
        {correctCount} of {targetWords.length} words found
      </p>

      <div className="flex justify-center">
        <Button type="button" onClick={() => onDone(correctCount)}>
          Done
        </Button>
      </div>
    </div>
  );
}

/** Phase 3 – Serial 7s subtraction game. */
function Serial7Phase({ onDone }: { onDone: (score: number) => void }) {
  const [round, setRound] = useState(0);
  const [currentNumber, setCurrentNumber] = useState(100);
  const [input, setInput] = useState("");
  const [results, setResults] = useState<
    { expected: number; given: number; correct: boolean }[]
  >([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const submitAnswer = () => {
    const answer = parseInt(input, 10);
    if (isNaN(answer)) return;

    const expected = currentNumber - 7;
    const correct = answer === expected;

    setResults((prev) => [...prev, { expected, given: answer, correct }]);
    // TICS protocol: next round subtracts from user's answer
    setCurrentNumber(answer);
    setInput("");
    setRound((r) => r + 1);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submitAnswer();
    }
  };

  useEffect(() => {
    if (round > 0 && round < 5) {
      inputRef.current?.focus();
    }
  }, [round]);

  const isFinished = round >= 5;
  const score = results.filter((r) => r.correct).length;

  return (
    <div className="space-y-6">
      <div className="text-center space-y-1">
        <h3 className="text-lg font-semibold">Quick Math</h3>
        <p className="text-sm text-muted-foreground">
          Starting from 100, keep subtracting 7. Let&apos;s see how you do!
        </p>
      </div>

      {!isFinished ? (
        <>
          <div className="text-center">
            <p className="text-3xl font-bold tabular-nums">{currentNumber}</p>
            <p className="text-sm text-muted-foreground mt-1">
              minus 7 = ?&ensp;(Round {round + 1} of 5)
            </p>
          </div>

          <div className="flex justify-center gap-2">
            <Input
              ref={inputRef}
              type="number"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-28 text-center text-lg"
              autoFocus
            />
            <Button type="button" size="sm" onClick={submitAnswer}>
              Go
            </Button>
          </div>

          {results.length > 0 && (
            <div className="flex justify-center gap-2">
              {results.map((r, i) => (
                <span
                  key={i}
                  className={`flex size-8 items-center justify-center rounded-full text-sm font-medium ${
                    r.correct
                      ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                      : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                  }`}
                >
                  {r.correct ? "\u2713" : "\u2717"}
                </span>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="text-center space-y-4">
          <div className="flex justify-center gap-2">
            {results.map((r, i) => (
              <div key={i} className="text-center">
                <span
                  className={`flex size-10 items-center justify-center rounded-full text-sm font-bold ${
                    r.correct
                      ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                      : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                  }`}
                >
                  {r.given}
                </span>
                {!r.correct && (
                  <p className="text-xs text-muted-foreground mt-1">
                    ({r.expected})
                  </p>
                )}
              </div>
            ))}
          </div>
          <p className="text-sm font-medium">
            {score} of 5 correct
          </p>
          <Button type="button" onClick={() => onDone(score)}>
            Continue
          </Button>
        </div>
      )}
    </div>
  );
}

/** Phase 4 – Countdown: click numbers 20→1 on a shuffled grid. */
function CountdownPhase({ onDone }: { onDone: (score: number) => void }) {
  const numbers = useMemo(() => shuffle(Array.from({ length: 20 }, (_, i) => i + 1)), []);
  const [nextExpected, setNextExpected] = useState(20);
  const [clicked, setClicked] = useState<Set<number>>(new Set());
  const [attempt, setAttempt] = useState(1);
  const [flash, setFlash] = useState<number | null>(null);
  const errorsRef = useRef(0);

  const handleClick = (n: number) => {
    if (n !== nextExpected) {
      errorsRef.current += 1;
      setFlash(n);
      setTimeout(() => setFlash(null), 400);
      return;
    }

    const newClicked = new Set(clicked).add(n);
    setClicked(newClicked);
    const newExpected = nextExpected - 1;
    setNextExpected(newExpected);

    // Check if this round is complete (user just clicked 1)
    if (newExpected === 0) {
      if (errorsRef.current === 0) {
        // Error-free: 2 points on first try, 1 on second
        onDone(attempt === 1 ? 2 : 1);
      } else if (attempt < 2) {
        // Had errors on first attempt — reset for second try
        setAttempt(2);
        setNextExpected(20);
        setClicked(new Set());
        errorsRef.current = 0;
      } else {
        // Had errors on second attempt too
        onDone(0);
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center space-y-1">
        <h3 className="text-lg font-semibold">Countdown Challenge</h3>
        <p className="text-sm text-muted-foreground">
          Click the numbers from <strong>20</strong> down to{" "}
          <strong>1</strong> in order.
          {attempt === 2 && " Second try — you can do it!"}
        </p>
      </div>

      <p className="text-center text-sm font-medium">
        Next: <span className="text-primary text-lg">{nextExpected > 0 ? nextExpected : "done!"}</span>
      </p>

      <div className="grid grid-cols-5 gap-2 max-w-xs mx-auto">
        {numbers.map((n) => {
          const done = clicked.has(n);
          const isFlash = flash === n;
          return (
            <Button
              key={n}
              type="button"
              variant={done ? "default" : "outline"}
              size="lg"
              disabled={done}
              onClick={() => handleClick(n)}
              className={`text-base font-bold tabular-nums transition-colors ${
                done
                  ? "bg-green-600 text-white hover:bg-green-600 dark:bg-green-700"
                  : isFlash
                    ? "bg-red-100 border-red-400 dark:bg-red-900/30"
                    : ""
              }`}
            >
              {n}
            </Button>
          );
        })}
      </div>
    </div>
  );
}

/** Phase 6 – Results summary. */
function ResultsPhase({
  imrc,
  dlrc,
  ser7,
  countdown,
}: {
  imrc: number;
  dlrc: number;
  ser7: number;
  countdown: number;
}) {
  const tr20 = imrc + dlrc;
  const cog27 = tr20 + ser7 + countdown;

  const rows = [
    { label: "Words recalled right away", value: imrc, max: 10 },
    { label: "Words recalled later", value: dlrc, max: 10 },
    { label: "Word memory — total", value: tr20, max: 20 },
    { label: "Quick math score", value: ser7, max: 5 },
    { label: "Countdown score", value: countdown, max: 2 },
    { label: "Overall thinking score", value: cog27, max: 27 },
  ];

  return (
    <div className="space-y-6">
      <div className="text-center space-y-1">
        <h3 className="text-lg font-semibold">Your Results</h3>
        <p className="text-sm text-muted-foreground">
          Here&apos;s how you did. Click &quot;Next&quot; below to continue.
        </p>
      </div>

      <div className="divide-y rounded-lg border">
        {rows.map((r) => (
          <div
            key={r.label}
            className="flex items-center justify-between px-4 py-3"
          >
            <span className="text-sm">{r.label}</span>
            <span className="font-semibold tabular-nums">
              {r.value}
              <span className="text-muted-foreground font-normal">
                /{r.max}
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main CognitionStep component
// ---------------------------------------------------------------------------

export default function CognitionStep({ form }: StepProps) {
  const { setValue, getValues } = form;
  const completedRef = useRef(false);

  // Pick a word list once on mount
  const words = useMemo(() => pickWords(), []);

  // Scores
  const [imrc, setImrc] = useState(0);
  const [dlrc, setDlrc] = useState(0);
  const [ser7, setSer7] = useState(0);
  const [countdown, setCountdown] = useState(0);

  // Check if games were already completed (re-entry)
  const initialPhase = useMemo<Phase>(() => {
    const existing = getValues();
    if (
      existing.IMRC > 0 ||
      existing.DLRC > 0 ||
      existing.SER7 > 0 ||
      existing.COG27 > 0
    ) {
      return "results";
    }
    return "memorize";
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [phase, setPhase] = useState<Phase>(initialPhase);

  // Pre-fill scores on re-entry
  useEffect(() => {
    if (initialPhase === "results") {
      const v = getValues();
      setImrc(v.IMRC);
      setDlrc(v.DLRC);
      setSer7(v.SER7);
      // Infer countdown from COG27
      const tr20 = v.IMRC + v.DLRC;
      setCountdown(v.COG27 - tr20 - v.SER7);
      completedRef.current = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Write scores to form when reaching results phase
  useEffect(() => {
    if (phase === "results" && !completedRef.current) {
      completedRef.current = true;
      const tr20 = imrc + dlrc;
      const cog27 = tr20 + ser7 + countdown;
      setValue("IMRC", imrc, { shouldValidate: true });
      setValue("DLRC", dlrc, { shouldValidate: true });
      setValue("TR20", tr20, { shouldValidate: true });
      setValue("SER7", ser7, { shouldValidate: true });
      setValue("COG27", cog27, { shouldValidate: true });
    }
  }, [phase, imrc, dlrc, ser7, countdown, setValue]);

  // Stable callbacks for phase transitions
  const handleMemorizeDone = useCallback(() => setPhase("immediateRecall"), []);

  const handleImmediateRecallDone = useCallback(
    (score: number) => {
      setImrc(score);
      setPhase("serial7");
    },
    [],
  );

  const handleSerial7Done = useCallback(
    (score: number) => {
      setSer7(score);
      setPhase("countdown");
    },
    [],
  );

  const handleCountdownDone = useCallback(
    (score: number) => {
      setCountdown(score);
      setPhase("delayedRecall");
    },
    [],
  );

  const handleDelayedRecallDone = useCallback(
    (score: number) => {
      setDlrc(score);
      setPhase("results");
    },
    [],
  );

  return (
    <FieldSet className="gap-4">
      <FieldLegend className="w-full border-b border-border pb-1 text-lg font-bold text-primary">
        Memory &amp; Thinking
      </FieldLegend>
      <FieldDescription>
        Let&apos;s play a few quick games to check your memory and thinking
        skills.
      </FieldDescription>

      {phase === "memorize" && (
        <MemorizePhase words={words} onDone={handleMemorizeDone} />
      )}

      {phase === "immediateRecall" && (
        <RecallPhase
          title="What Do You Remember?"
          description="Type the words you just saw, one at a time."
          targetWords={words}
          onDone={handleImmediateRecallDone}
        />
      )}

      {phase === "serial7" && <Serial7Phase onDone={handleSerial7Done} />}

      {phase === "countdown" && (
        <CountdownPhase onDone={handleCountdownDone} />
      )}

      {phase === "delayedRecall" && (
        <RecallPhase
          title="Do You Still Remember?"
          description="Earlier we showed you some words — how many can you recall now?"
          targetWords={words}
          onDone={handleDelayedRecallDone}
        />
      )}

      {phase === "results" && (
        <ResultsPhase
          imrc={imrc}
          dlrc={dlrc}
          ser7={ser7}
          countdown={countdown}
        />
      )}
    </FieldSet>
  );
}
