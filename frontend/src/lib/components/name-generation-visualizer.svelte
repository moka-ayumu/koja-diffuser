<script lang="ts">
	import { untrack } from 'svelte';
	import TensorView from '$lib/components/tensor-view.svelte';
	import type { GenerateName } from '$lib/api.svelte';
	import { detectLang, kanaToHangul } from '$lib/language';

	type Language = 'hangul' | 'hiragana';
	type Tensor = number[][];
	type VisualStep =
		| { kind: 'input'; label: string }
		| { kind: 'encoded'; label: string; ids: number[] }
		| { kind: 'tensor'; label: string; tensor: Tensor }
		| { kind: 'diffusion'; label: string; tensors: Tensor[] }
		| { kind: 'decoded'; label: string; ids: number[] }
		| { kind: 'result'; label: string; ids: number[]; result: string };
	type TimelineUnit = {
		stepIndex: number;
		diffusionIndex?: number;
	};
	type Props = {
		name: string;
		language: Language;
		api?: GenerateName;
		class?: string;
	};

	const minStepDelay = 500;
	const activeCellSize = 8;
	const historyCellSize = 4;

	let { name, language, api, class: className = '' }: Props = $props();

	let displayIndex = $state(0);
	let userScrubbing = $state(false);
	let stepColumn: HTMLDivElement | undefined;
	let lastApi: GenerateName | undefined;
	let autoplayTimer: number | undefined;
	let lastWheelAt = 0;

	const trimmedName = $derived(name.trim());
	const characters = $derived(Array.from(trimmedName));
	const slotCount = $derived(language === 'hiragana' ? 20 : 10);
	const slotIndexes = $derived(Array.from({ length: slotCount }, (_, index) => index));
	const stepList = (() => {
		const items = $state<VisualStep[]>([]);
		const diffusionTensors = $state<Tensor[]>([]);
		let currentApi: GenerateName | undefined;
		let currentName = trimmedName;

		function reset(nextApi: GenerateName | undefined, inputLabel: string) {
			currentApi = nextApi;
			diffusionTensors.length = 0;
			items.length = 0;
			items.push({ kind: 'input', label: inputLabel });
		}

		$effect(() => {
			const inputLabel = trimmedName ? 'input slots' : 'waiting';

			if (trimmedName !== currentName) {
				untrack(() => {
					currentName = trimmedName;
					reset(api, inputLabel);
				});
				return;
			}

			if (items.length === 0 || api !== currentApi) reset(api, inputLabel);
			else if (items[0]?.kind === 'input') items[0].label = inputLabel;

			if (api?.name !== trimmedName) return;

			if (api?.encoded?.ids) {
				if (items.length === 1)
					items.push({ kind: 'encoded', label: 'encoded ids', ids: api.encoded!.ids });
			}

			if (api?.encoded?.tensor) {
				if (items.length === 2)
					items.push({ kind: 'tensor', label: 'encoded tensor', tensor: api.encoded!.tensor });
			}

			if (api?.bridge_guide) {
				if (items.length === 3)
					items.push({ kind: 'tensor', label: 'encoded guide', tensor: api.bridge_guide! });
			}

			for (let index = untrack(() => diffusionTensors.length); index < api.ddim.length; index++) {
				const tensor = api.ddim[index].x;

				diffusionTensors.push(tensor);
			}

			if (items.length === 4 && diffusionTensors.length > 0) {
				items.push({ kind: 'diffusion', label: 'diffusion', tensors: diffusionTensors });
			}

			if (api?.decoded?.ids) {
				if (items.length === 5)
					items.push({ kind: 'decoded', label: 'decoded ids', ids: api.decoded!.ids });
			}

			if (api?.decoded?.result !== undefined && api.decoded.ids) {
				if (items.length === 6) {
					items.push({
						kind: 'result',
						label: 'result',
						ids: api.decoded!.ids,
						result: api.decoded!.result
					});
				}
			}
		});

		return {
			get current() {
				return items;
			}
		};
	})();
	const timelineUnits = $derived.by<TimelineUnit[]>(() =>
		stepList.current.flatMap((step, stepIndex) => {
			if (step.kind !== 'diffusion') return [{ stepIndex }];

			return step.tensors.map((_, diffusionIndex) => ({ stepIndex, diffusionIndex }));
		})
	);
	const safeDisplayIndex = $derived(clamp(displayIndex, 0, Math.max(0, timelineUnits.length - 1)));
	const activeUnit = $derived(timelineUnits[safeDisplayIndex]);
	const activeIndex = $derived(activeUnit?.stepIndex ?? 0);
	const activeStep = $derived(stepList.current[activeIndex] ?? stepList.current[0]);
	const activeDiffusionIndex = $derived(activeUnit?.diffusionIndex ?? 0);
	const visibleSteps = $derived(stepList.current.slice(0, activeIndex + 1));
	const phaseLabel = $derived(getPhaseLabel(activeStep, activeDiffusionIndex));
	const sliderMax = $derived(Math.max(0, timelineUnits.length - 1));
	const canUseTimeline = $derived(timelineUnits.length > 1);

	$effect(() => {
		if (api === lastApi) return;

		lastApi = api;
		userScrubbing = false;
		displayIndex = 0;
	});

	$effect(() => {
		if (displayIndex <= sliderMax) return;

		displayIndex = sliderMax;
	});

	$effect(() => {
		if (autoplayTimer) {
			window.clearTimeout(autoplayTimer);
			autoplayTimer = undefined;
		}

		if (userScrubbing) return;
		if (displayIndex >= timelineUnits.length - 1) return;

		autoplayTimer = window.setTimeout(() => {
			displayIndex = Math.min(displayIndex + 1, timelineUnits.length - 1);
			autoplayTimer = undefined;
		}, minStepDelay);

		return () => {
			if (autoplayTimer) {
				window.clearTimeout(autoplayTimer);
				autoplayTimer = undefined;
			}
		};
	});

	$effect(() => {
		if (!stepColumn) return;

		// eslint-disable-next-line @typescript-eslint/no-unused-expressions
		[displayIndex, activeIndex];

		window.requestAnimationFrame(() => {
			const target = stepColumn?.querySelector<HTMLElement>('[data-active="true"]');

			target?.scrollIntoView({
				block: 'center',
				behavior: 'smooth'
			});
		});
	});

	function clamp(n: number, min: number, max: number) {
		return Math.max(min, Math.min(max, n));
	}

	function getPhaseLabel(step: VisualStep | undefined, diffusionIndex: number) {
		if (!step) return 'waiting';
		if (step.kind === 'diffusion')
			return `diffusion step ${diffusionIndex + 1} / ${step.tensors.length}`;

		return step.label;
	}

	function getStepIds(step: VisualStep | undefined) {
		if (!step) return undefined;
		if (step.kind === 'encoded' || step.kind === 'decoded' || step.kind === 'result')
			return step.ids;

		return undefined;
	}

	function handleSliderInput(event: Event) {
		const input = event.currentTarget as HTMLInputElement;

		userScrubbing = true;
		displayIndex = Number(input.value);
	}

	function resumeAutoplay() {
		userScrubbing = false;
	}

	function getStepSlotText(step: VisualStep, index: number) {
		const ids = getStepIds(step);
		const id = ids?.[index];

		if (Number.isFinite(id)) return String(id);
		if (step.kind === 'input') return characters[index] ?? '';

		return '';
	}

	function handleWheel(event: WheelEvent) {
		if (timelineUnits.length <= 1) return;

		event.preventDefault();

		const now = performance.now();

		if (now - lastWheelAt < 120) return;
		if (Math.abs(event.deltaY) < 8) return;

		lastWheelAt = now;
		userScrubbing = true;
		displayIndex = clamp(displayIndex + (event.deltaY > 0 ? 1 : -1), 0, timelineUnits.length - 1);
	}
</script>

<section
	class={`relative flex h-screen flex-1 items-center justify-center overflow-hidden bg-background ${className}`}
	aria-label="Name generation visualizer"
	onwheel={handleWheel}
>
	<div
		class="absolute top-6 right-7 z-40 rounded-full border border-border bg-background/80 px-2.5 py-1.5 text-[11px] leading-none text-muted-foreground uppercase tabular-nums backdrop-blur-md max-md:top-4 max-md:right-4"
	>
		{phaseLabel}
	</div>

	<div
		class="relative z-20 h-full w-[min(780px,calc(100vw-360px))] min-w-70 overflow-hidden py-16 pb-20 max-md:w-[min(100%-32px,560px)]"
	>
		<div
			class="step-column flex h-full flex-col gap-4 overflow-y-auto overscroll-contain"
			bind:this={stepColumn}
		>
			<div class="shrink-0 basis-[42vh]"></div>
			{#each visibleSteps as step, index (`${index}-${step.label}`)}
				<div
					class={`flex w-full shrink-0 origin-center flex-col items-center gap-2.5 filter transition-[opacity,transform,filter] duration-300 ease-out ${
						index === activeIndex
							? 'scale-100 opacity-100 saturate-100'
							: 'scale-[0.96] opacity-60 saturate-[0.88]'
					}`}
					data-active={index === activeIndex}
				>
					<div
						class={`rounded-full border border-border bg-background/75 px-2.5 py-1 text-[10px] leading-none uppercase tabular-nums backdrop-blur-md ${
							index === activeIndex ? 'text-foreground' : 'text-muted-foreground'
						}`}
					>
						{step.label}
					</div>

					{#if step.kind === 'input'}
						<div
							class={`max-w-[min(560px,100%)] overflow-hidden text-center leading-[1.05] font-semibold text-ellipsis whitespace-nowrap ${
								trimmedName
									? 'text-[clamp(30px,6vw,70px)] text-foreground'
									: 'text-2xl font-medium text-muted-foreground'
							}`}
						>
							{trimmedName || 'waiting'}
						</div>
						{#if detectLang(trimmedName) === 'hiragana'}
							<p>{kanaToHangul(trimmedName)}</p>
						{/if}
						<div
							class="grid w-[min(620px,100%)] gap-2 transition-[opacity,transform] duration-300"
							style={`grid-template-columns: repeat(${slotCount}, minmax(0, 1fr));`}
						>
							{#each slotIndexes as slotIndex (slotIndex)}
								{@const text = getStepSlotText(step, slotIndex)}
								<div
									class={`relative flex aspect-square min-w-0 items-center justify-center overflow-hidden rounded-lg border border-border text-[clamp(10px,1.45vw,17px)] leading-none font-semibold tabular-nums ${
										text
											? 'bg-background text-foreground shadow-[0_12px_32px_rgba(24,24,27,0.05)]'
											: 'bg-muted text-transparent shadow-none'
									}`}
								>
									<span class="block max-w-full overflow-hidden text-ellipsis whitespace-nowrap"
										>{text}</span
									>
								</div>
							{/each}
						</div>
					{:else if step.kind === 'encoded' || step.kind === 'decoded'}
						<div
							class="grid w-[min(620px,100%)] gap-2 transition-[opacity,transform] duration-300"
							style={`grid-template-columns: repeat(${slotCount}, minmax(0, 1fr));`}
						>
							{#each slotIndexes as slotIndex (slotIndex)}
								{@const text = getStepSlotText(step, slotIndex)}
								<div
									class={`relative flex aspect-square min-w-0 items-center justify-center overflow-hidden rounded-lg border text-[clamp(10px,1.45vw,17px)] leading-none font-semibold tabular-nums ${
										text
											? `${step.kind === 'decoded' ? 'border-[#e8a7b8]' : 'border-[#afcbff]'} bg-background text-foreground shadow-[0_12px_32px_rgba(24,24,27,0.05)]`
											: 'border-border bg-muted text-transparent shadow-none'
									}`}
								>
									<span class="block max-w-full overflow-hidden text-ellipsis whitespace-nowrap"
										>{text}</span
									>
								</div>
							{/each}
						</div>
					{:else if step.kind === 'tensor'}
						<TensorView
							tensors={[step.tensor]}
							animated={index === activeIndex && !userScrubbing}
							interactive={index === activeIndex}
							cellSize={index === activeIndex ? activeCellSize : historyCellSize}
							maxWidth={index === activeIndex ? 680 : 560}
							maxHeight={index === activeIndex ? 360 : 140}
							duration={Math.max(260, minStepDelay - 160)}
							opacity={index === activeIndex ? 1 : 0.92}
						/>
					{:else if step.kind === 'diffusion'}
						<TensorView
							tensors={step.tensors}
							frameIndex={index === activeIndex ? activeDiffusionIndex : step.tensors.length - 1}
							animated={index === activeIndex && !userScrubbing}
							interactive={index === activeIndex}
							cellSize={index === activeIndex ? activeCellSize : historyCellSize}
							maxWidth={index === activeIndex ? 680 : 560}
							maxHeight={index === activeIndex ? 360 : 140}
							duration={Math.max(260, minStepDelay - 160)}
							opacity={index === activeIndex ? 1 : 0.92}
						/>
						<div class="text-[11px] leading-none text-muted-foreground tabular-nums">
							step {index === activeIndex ? activeDiffusionIndex + 1 : step.tensors.length} / {step
								.tensors.length}
						</div>
					{:else if step.kind === 'result'}
						<div
							class="grid w-[min(620px,100%)] gap-2 opacity-70 transition-[opacity,transform] duration-300"
							style={`grid-template-columns: repeat(${slotCount}, minmax(0, 1fr));`}
						>
							{#each slotIndexes as slotIndex (slotIndex)}
								{@const text = getStepSlotText(step, slotIndex)}
								<div
									class={`relative flex aspect-square min-w-0 items-center justify-center overflow-hidden rounded-lg border text-[clamp(10px,1.45vw,17px)] leading-none font-semibold tabular-nums ${
										text
											? 'border-[#e8a7b8] bg-background text-foreground shadow-[0_12px_32px_rgba(24,24,27,0.05)]'
											: 'border-border bg-muted text-transparent shadow-none'
									}`}
								>
									<span class="block max-w-full overflow-hidden text-ellipsis whitespace-nowrap"
										>{text}</span
									>
								</div>
							{/each}
						</div>
						<div
							class="overflow-wrap-anywhere max-w-[min(640px,92%)] text-center text-[clamp(42px,8vw,96px)] leading-[1.02] font-bold text-foreground"
						>
							{step.result}
						</div>
						{#if detectLang(step.result) === 'hiragana'}
							<p>{kanaToHangul(step.result)}</p>
						{/if}
					{/if}
				</div>
			{/each}
			<div class="shrink-0 basis-[42vh]"></div>
		</div>
	</div>

	<div
		class="absolute right-7 bottom-6 left-7 z-40 grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-2.5 max-md:right-4 max-md:bottom-4 max-md:left-4"
	>
		<input
			class="w-full accent-[#afcbff]"
			type="range"
			min="0"
			max={sliderMax}
			step="1"
			value={safeDisplayIndex}
			disabled={!canUseTimeline}
			aria-label="generation step"
			oninput={handleSliderInput}
		/>
		<button
			class="rounded-lg border border-border bg-background/80 px-2.5 py-2 text-xs leading-none text-foreground backdrop-blur-md disabled:cursor-not-allowed disabled:opacity-40"
			type="button"
			disabled={!canUseTimeline || safeDisplayIndex >= sliderMax}
			onclick={resumeAutoplay}
		>
			Auto
		</button>
		<div class="min-w-13 text-right text-xs text-muted-foreground tabular-nums">
			{safeDisplayIndex + 1} / {timelineUnits.length}
		</div>
	</div>
</section>

<style>
	/* Tailwind does not include a cross-browser scrollbar-none utility in this setup. */
	.step-column {
		scrollbar-width: none;
	}
	.step-column::-webkit-scrollbar {
		display: none;
	}
</style>
