import { fetchEventSource } from '@microsoft/fetch-event-source';

interface Ddim {
	x: number[][];
}

type StageKey = 'encoded' | 'decoded' | 'bridge_guide' | 'ddim';

type StageValue<Key extends StageKey> = Key extends 'ddim' ? Ddim : GenerateName[Key];

type StageMap = {
	[Key in StageKey]: {
		name: Key;
		i: number;
		stage: NonNullable<StageValue<Key>>;
	};
};

type Stage<Key extends StageKey = StageKey> = StageMap[Key];

export interface GenerateOptions {
	sampling_mode?: 'greedy' | 'sample';
	seed?: number;
	temperature?: number;
	top_k?: number;
	top_p?: number;
}

export class GenerateName {
	name = $state('');
	age = $state(0);
	options = $state<GenerateOptions>({});
	info = $state<{ seed: number; type: 'ko_to_ja' | 'ja_to_ko' }>();
	encoded = $state<{ tensor: number[][]; ids: number[] }>();
	decoded = $state<{ result: string; ids: number[] }>();
	bridge_guide = $state<number[][]>();
	ddim = $state<Ddim[]>([]);
	//
	current = 0;
	stages = [
		'encoded/0',
		'bridge_guide/0',
		...Array.from({ length: 6 }, (_, i) => `ddim/${i}` as const),
		'decoded/0'
	] as const;

	constructor(name: string, age: number, options: GenerateOptions) {
		this.name = name;
		this.age = age;
		this.options = options;
		this.fetch();
	}

	async fetch() {
		await fetchEventSource('/api/stream', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				name: this.name,
				age: this.age,
				...this.options
			}),
			onmessage: (event) => {
				const json = JSON.parse(event.data);

				switch (event.event) {
					case 'info':
						this.info = json;
						break;

					case 'encoded':
						this.encoded = { tensor: json.guide[0], ids: json.names_ids[0] };
						break;

					case 'ddim.init_noise':
						this.bridge_guide = json.guide_encoded[0];
						this.ddim.push({ x: json.x[0] });
						break;

					case 'ddim.step':
						this.ddim.push({ x: json.x[0] });
						break;

					case 'decoded':
						this.decoded = { result: json.result[0], ids: json.names_ids[0] };
						break;

					default:
						break;
				}
			},
			onerror(error) {
				console.error('SSE error:', error);
				throw error;
			}
		});
	}

	loadedStages = $derived.by(() => {
		const res: Stage[] = [];
		for (let i = 0; i < this.stages.length; i++) {
			const [name, index_raw] = this.stages[i].split('/') as [StageKey, string];
			const index = Number(index_raw);
			if (this[name] !== undefined) {
				if (name === 'ddim') {
					if (this[name].length > index) {
						res.push({
							name,
							i: index,
							stage: this[name][index]
						});
						continue;
					}
				} else {
					res.push({
						name,
						i: index,
						stage: this[name]
					} as Stage);
					continue;
				}
			}
			break;
		}
		return res;
	});
}
