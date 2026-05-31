<script lang="ts">
	import NameGenerationVisualizer from '$components/name-generation-visualizer.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import * as InputGroup from '$lib/components/ui/input-group';
	import * as Label from '$lib/components/ui/label';
	import { GenerateName, type GenerateOptions } from '$lib/api.svelte';
	import { IterationCwIcon, LanguagesIcon } from '@lucide/svelte';
	import { Slider } from '$lib/components/ui/slider';
	import { detectLang } from '$lib/language';
	import * as Tabs from '$components/ui/tabs';

	let inputValue = $state('');

	const langValue = $derived(detectLang(inputValue));
	let currentApi = $state<GenerateName>();

	let prevAgeSliderValue = $state([0, 10]);
	let ageSliderValue = $state([0, 10]);
	const DEFAULT_OPTIONS: GenerateOptions = {
		sampling_mode: 'sample',
		temperature: 0.8,
		top_k: 20,
		top_p: 0.9,
		seed: undefined
	} satisfies GenerateOptions;
	let options = $state<GenerateOptions>({ ...DEFAULT_OPTIONS });
	function setAge(i: number) {
		ageSliderValue = [i, i + 10];
	}

	$effect(() => {
		const diffValue = ageSliderValue[1] - ageSliderValue[0];
		if (diffValue >= 20) {
			const weight = (Math.floor(diffValue / 10) - 1) * 10;
			console.log(ageSliderValue[0], ageSliderValue[1], weight);
			if (ageSliderValue[0] == prevAgeSliderValue[0]) {
				setAge(ageSliderValue[0] + weight);
			} else {
				setAge(ageSliderValue[0] - weight);
			}
		} else if (diffValue === 0) {
			if (ageSliderValue[0] === 0) {
				ageSliderValue = [0, 10];
			} else if (ageSliderValue[0] === 9) {
				ageSliderValue = [90, 100];
			} else if (ageSliderValue[0] == prevAgeSliderValue[0]) {
				setAge(ageSliderValue[0] + 10);
			} else {
				setAge(ageSliderValue[0] - 10);
			}
		}
		prevAgeSliderValue = ageSliderValue;
	});

	function generateName() {
		const name = inputValue.trim();

		if (!name) return;

		currentApi = new GenerateName(name, Math.floor(ageSliderValue[0] / 10), options);
	}
</script>

{#snippet reset_button(key: keyof GenerateOptions)}
	{#if options[key] !== DEFAULT_OPTIONS[key]}
		<Button
			size="icon-xs"
			variant="ghost"
			onclick={() => (options[key] = DEFAULT_OPTIONS[key] as never)}
		>
			<IterationCwIcon />
		</Button>
	{/if}
{/snippet}

<div class="flex h-screen w-screen overflow-hidden">
	<Card.Root class="z-10 h-screen w-80 shrink-0 rounded-none border-r shadow-none" size="sm">
		<Card.Header>
			<Card.Title>이름 생성기</Card.Title>
			<Card.Description>인코딩, 확산, 디코딩 과정을 시각화합니다.</Card.Description>
		</Card.Header>
		<Card.Content class="flex flex-col gap-2">
			<InputGroup.Root>
				<InputGroup.Input
					id="name"
					placeholder={langValue === 'hiragana' ? 'ひらがな' : '이름'}
					bind:value={inputValue}
					onkeydown={(event) => {
						if (event.key === 'Enter') generateName();
					}}
				/>
				<InputGroup.Addon align="block-start">
					<Label.Root for="name" class="text-foreground">이름</Label.Root>
					<Label.Root for="name" class="ms-auto">
						<LanguagesIcon size={14} />
						{langValue === 'hangul' ? 'Korean' : 'Japanese'}
					</Label.Root>
				</InputGroup.Addon>
			</InputGroup.Root>
			<InputGroup.Root>
				<div class="h-16 w-11/12 pt-10">
					<Slider type="multiple" min={0} max={100} step={10} tick bind:value={ageSliderValue} />
				</div>
				<InputGroup.Addon align="block-start">
					<Label.Root for="age" class="text-foreground">연령대</Label.Root>
				</InputGroup.Addon>
			</InputGroup.Root>
			<InputGroup.Root>
				<InputGroup.Input id="seed" placeholder="-1" type="number" bind:value={options.seed} />
				<InputGroup.Addon align="block-start">
					<Label.Root for="seed" class="text-foreground">시드</Label.Root>
				</InputGroup.Addon>
			</InputGroup.Root>
			<Tabs.Root bind:value={options.sampling_mode}>
				<InputGroup.Root>
					<Tabs.Content value="sample" class="w-full">
						<InputGroup.Root>
							<div class="h-16 w-11/12 pt-10">
								<Slider
									type="single"
									min={0.1}
									max={2.0}
									step={0.05}
									tick={({ value }) => value === 0.1 || value % 0.5 === 0}
									bind:value={options.temperature}
								/>
							</div>
							<InputGroup.Addon align="block-start" class="h-6">
								<Label.Root class="text-foreground">Temperature</Label.Root>
								<Label.Root for="name" class="ms-auto">
									{options.temperature}
								</Label.Root>
								{@render reset_button('temperature')}
							</InputGroup.Addon>
						</InputGroup.Root>
						<InputGroup.Root>
							<div class="h-16 w-11/12 pt-10">
								<Slider
									type="single"
									min={1}
									max={100}
									step={1}
									tick={({ value }) => value === 1 || value % 10 === 0}
									bind:value={options.top_k}
								/>
							</div>
							<InputGroup.Addon align="block-start" class="h-6">
								<Label.Root class="text-foreground">Top-K</Label.Root>
								<Label.Root for="name" class="ms-auto">
									{options.top_k}
								</Label.Root>
								{@render reset_button('top_k')}
							</InputGroup.Addon>
						</InputGroup.Root>
						<InputGroup.Root>
							<div class="h-16 w-11/12 pt-10">
								<Slider
									type="single"
									min={0.1}
									max={1.0}
									step={0.01}
									tick={({ value }) => value === 1 || String(value).length === 3}
									bind:value={options.top_p}
								/>
							</div>
							<InputGroup.Addon align="block-start" class="h-6">
								<Label.Root class="text-foreground">Top-P</Label.Root>
								<Label.Root for="name" class="ms-auto">
									{options.top_p}
								</Label.Root>
								{@render reset_button('top_p')}
							</InputGroup.Addon>
						</InputGroup.Root>
					</Tabs.Content>
					<Tabs.Content value="greedy"></Tabs.Content>
					<InputGroup.Addon align="block-start">
						<Label.Root for="sampling" class="text-foreground">샘플링</Label.Root>
						<Tabs.List class="ms-auto">
							<Tabs.Trigger value="sample">Sample</Tabs.Trigger>
							<Tabs.Trigger value="greedy">Greedy</Tabs.Trigger>
						</Tabs.List>
					</InputGroup.Addon>
				</InputGroup.Root>
			</Tabs.Root>
		</Card.Content>
		<Card.Footer>
			<Button class="w-full" disabled={!inputValue.trim()} onclick={generateName}>Generate</Button>
		</Card.Footer>
	</Card.Root>

	<NameGenerationVisualizer name={inputValue} language={langValue} api={currentApi} />
</div>
