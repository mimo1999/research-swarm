"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useEffect, useRef, useState } from "react";
import { Zap, Scale, Microscope, Monitor, Cloud, Paperclip, ChevronRight, SlidersHorizontal, type LucideIcon } from "lucide-react";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { useConfigOptions } from "@/lib/research/useConfigOptions";
import { useOllamaStatus } from "@/lib/research/useOllamaStatus";
import type { ResearchSubmitBody } from "@/lib/research/useResearchRun";

const schema = z.object({
  topic: z.string().min(1, "Required"),
  audience: z.enum(["general", "technical", "academic", "executive"]),
  provider: z.enum(["ollama", "anthropic", "openai"]),
  model: z.string().min(1, "Required"),
  depth: z.enum(["shallow", "standard", "deep"]),
  maxSources: z.number().min(3).max(30),
  hitlEnabled: z.boolean(),
  ollamaUrl: z.string().optional(),
  ollamaDeployment: z.enum(["local", "cloud"]).optional(),
  urls: z.string().optional(),
});
type FormValues = z.infer<typeof schema>;

const DEPTH_OPTIONS: { value: string; label: string; hint: string; icon: LucideIcon }[] = [
  { value: "shallow", label: "Shallow", hint: "Fastest, 1 sub-question", icon: Zap },
  { value: "standard", label: "Standard", hint: "4 sub-questions", icon: Scale },
  { value: "deep", label: "Deep", hint: "6 sub-questions", icon: Microscope },
];

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function QueryForm({
  onSubmit,
}: {
  onSubmit: (body: ResearchSubmitBody, files: File[], urls: string[]) => void;
}) {
  const options = useConfigOptions();
  const [files, setFiles] = useState<File[]>([]);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      topic: "",
      audience: "technical",
      provider: "ollama",
      model: "",
      depth: "shallow",
      maxSources: 10,
      hitlEnabled: true,
      ollamaDeployment: "local",
      ollamaUrl: "",
      urls: "",
    },
  });

  // Fill provider/model/max-sources defaults once options load, without
  // clobbering anything the user has already typed.
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (!options || hydratedRef.current) return;
    hydratedRef.current = true;
    form.setValue("provider", options.defaults.provider as FormValues["provider"]);
    form.setValue("model", options.defaults.model);
    form.setValue("maxSources", options.defaults.max_sources);
    form.setValue("ollamaUrl", options.defaults.ollama_url);
    form.setValue(
      "ollamaDeployment",
      options.defaults.ollama_deployment as FormValues["ollamaDeployment"]
    );
  }, [options, form]);

  const provider = form.watch("provider");
  const ollamaDeployment = form.watch("ollamaDeployment");
  const ollamaUrl = form.watch("ollamaUrl");
  const model = form.watch("model");
  const depth = form.watch("depth");

  const ollamaStatus = useOllamaStatus({
    url: ollamaUrl,
    model,
    deployment: ollamaDeployment,
    enabled: provider === "ollama",
  });

  // A one-line recap of what's behind the collapsed panel, so picking the
  // defaults and moving on doesn't mean flying blind.
  const advancedSummary = [capitalize(provider), model || null, `${capitalize(depth)} depth`]
    .filter(Boolean)
    .join(" · ");

  function handleSubmit(values: FormValues) {
    const urls = (values.urls ?? "")
      .split("\n")
      .map((u) => u.trim())
      .filter(Boolean);
    onSubmit(
      {
        topic: values.topic.trim(),
        audience: values.audience,
        depth: values.depth,
        max_sources: values.maxSources,
        provider: values.provider,
        model: values.model,
        ollama_url: values.provider === "ollama" ? values.ollamaUrl : null,
        ollama_deployment: values.provider === "ollama" ? values.ollamaDeployment : null,
        hitl_enabled: values.hitlEnabled,
      },
      files,
      urls
    );
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
        <FormField
          control={form.control}
          name="topic"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Research topic</FormLabel>
              <FormControl>
                <Input placeholder="e.g. Impact of large language models on drug discovery" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="audience"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Audience</FormLabel>
              <Select onValueChange={field.onChange} value={field.value}>
                <FormControl>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {["general", "technical", "academic", "executive"].map((a) => (
                    <SelectItem key={a} value={a}>
                      {capitalize(a)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FormItem>
          )}
        />

        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="flex w-full items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
            >
              <ChevronRight className={cn("size-4 transition-transform", advancedOpen && "rotate-90")} />
              <SlidersHorizontal className="size-3.5" />
              Advanced options
              {!advancedOpen && <span className="text-xs">({advancedSummary})</span>}
            </button>
          </CollapsibleTrigger>

          <CollapsibleContent className="space-y-6 pt-4 data-[state=open]:animate-in data-[state=open]:fade-in data-[state=open]:slide-in-from-top-1 data-[state=open]:duration-200">
            <Card>
              <CardContent className="pt-4 space-y-4">
                <FormField
                  control={form.control}
                  name="provider"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Provider</FormLabel>
                      <RadioGroup onValueChange={field.onChange} value={field.value} className="flex gap-4">
                        {(options?.providers ?? ["ollama", "anthropic", "openai"]).map((p) => (
                          <div key={p} className="flex items-center gap-2">
                            <RadioGroupItem value={p} id={`provider-${p}`} />
                            <Label htmlFor={`provider-${p}`} className="font-normal capitalize">
                              {p}
                            </Label>
                          </div>
                        ))}
                      </RadioGroup>
                    </FormItem>
                  )}
                />

                {provider === "anthropic" && (
                  <ModelSelect control={form.control} models={options?.models.anthropic ?? []} />
                )}
                {provider === "openai" && (
                  <ModelSelect control={form.control} models={options?.models.openai ?? []} />
                )}
                {provider === "ollama" && (
                  <div className="space-y-3">
                    <FormField
                      control={form.control}
                      name="ollamaDeployment"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Deployment</FormLabel>
                          <RadioGroup onValueChange={field.onChange} value={field.value} className="flex gap-4">
                            <div className="flex items-center gap-2">
                              <RadioGroupItem value="local" id="deploy-local" />
                              <Label htmlFor="deploy-local" className="font-normal flex items-center gap-1.5">
                                <Monitor className="size-3.5 text-muted-foreground" /> Local
                              </Label>
                            </div>
                            <div className="flex items-center gap-2">
                              <RadioGroupItem value="cloud" id="deploy-cloud" />
                              <Label htmlFor="deploy-cloud" className="font-normal flex items-center gap-1.5">
                                <Cloud className="size-3.5 text-muted-foreground" /> Cloud
                              </Label>
                            </div>
                          </RadioGroup>
                        </FormItem>
                      )}
                    />

                    {ollamaDeployment === "cloud" ? (
                      <FormField
                        control={form.control}
                        name="model"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Cloud model</FormLabel>
                            <Select onValueChange={field.onChange} value={field.value}>
                              <FormControl>
                                <SelectTrigger className="w-full">
                                  <SelectValue placeholder="Select a model" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                {(options?.models.ollama_cloud ?? []).map((m) => (
                                  <SelectItem key={m} value={m}>
                                    {m}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </FormItem>
                        )}
                      />
                    ) : (
                      <>
                        <FormField
                          control={form.control}
                          name="model"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Model name</FormLabel>
                              <FormControl>
                                <Input placeholder="e.g. gemma4:4b" {...field} />
                              </FormControl>
                            </FormItem>
                          )}
                        />
                        <FormField
                          control={form.control}
                          name="ollamaUrl"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Ollama URL</FormLabel>
                              <FormControl>
                                <Input placeholder="http://localhost:11434" {...field} />
                              </FormControl>
                            </FormItem>
                          )}
                        />
                      </>
                    )}

                    {ollamaStatus && (
                      <Alert variant={ollamaStatus.reachable ? "default" : "destructive"}>
                        <AlertDescription>{ollamaStatus.message}</AlertDescription>
                      </Alert>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4 space-y-4">
                <FormField
                  control={form.control}
                  name="depth"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Depth</FormLabel>
                      <RadioGroup onValueChange={field.onChange} value={field.value} className="flex flex-col gap-2">
                        {DEPTH_OPTIONS.map(({ value, label, hint, icon: Icon }) => (
                          <div key={value} className="flex items-center gap-2">
                            <RadioGroupItem value={value} id={`depth-${value}`} />
                            <Label htmlFor={`depth-${value}`} className="font-normal flex items-center gap-1.5">
                              <Icon className="size-3.5 text-muted-foreground" />
                              {label}
                              <span className="text-muted-foreground">({hint})</span>
                            </Label>
                          </div>
                        ))}
                      </RadioGroup>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="maxSources"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max sources: {field.value}</FormLabel>
                      <Slider
                        min={3}
                        max={30}
                        step={1}
                        value={[field.value]}
                        onValueChange={([v]) => field.onChange(v)}
                      />
                    </FormItem>
                  )}
                />
              </CardContent>
            </Card>

            <FormField
              control={form.control}
              name="hitlEnabled"
              render={({ field }) => (
                <FormItem className="flex items-center justify-between">
                  <FormLabel>Pause before writing (Human-in-the-Loop)</FormLabel>
                  <FormControl>
                    <Switch checked={field.value} onCheckedChange={field.onChange} />
                  </FormControl>
                </FormItem>
              )}
            />

            <div className="space-y-2">
              <Label className="flex items-center gap-1.5">
                <Paperclip className="size-3.5 text-muted-foreground" /> Documents
              </Label>
              <Input
                type="file"
                accept=".pdf"
                multiple
                onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              />
              <FormField
                control={form.control}
                name="urls"
                render={({ field }) => (
                  <Textarea
                    placeholder={"https://example.com/paper\nhttps://arxiv.org/abs/..."}
                    {...field}
                  />
                )}
              />
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Button type="submit" size="lg" className="w-full">
          Start research
        </Button>
      </form>
    </Form>
  );
}

function ModelSelect({ control, models }: { control: any; models: string[] }) {
  return (
    <FormField
      control={control}
      name="model"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Model name</FormLabel>
          <Select onValueChange={field.onChange} value={field.value}>
            <FormControl>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              {models.map((m) => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FormItem>
      )}
    />
  );
}
