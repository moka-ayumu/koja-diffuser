<script lang="ts">
	import { onDestroy, onMount, untrack } from 'svelte';

	type Tensor = number[][];
	type Rgb = {
		r: number;
		g: number;
		b: number;
	};

	type Props = {
		tensors: Tensor[];
		frameIndex?: number;
		cellSize?: number;
		maxWidth?: number;
		maxHeight?: number;
		duration?: number;
		animated?: boolean;
		interactive?: boolean;
		opacity?: number;
		class?: string;
	};

	const palette = [
		hexToRgb('#c7b7e8'),
		hexToRgb('#afcbff'),
		hexToRgb('#a8dadc'),
		hexToRgb('#b7d7a8'),
		hexToRgb('#e6d3a3'),
		hexToRgb('#f0bfa7'),
		hexToRgb('#e8a7b8'),
		hexToRgb('#c9b6e4')
	];

	let {
		tensors,
		frameIndex = 0,
		cellSize = 8,
		maxWidth = 680,
		maxHeight = 360,
		duration = 760,
		animated = true,
		interactive = true,
		opacity = 1,
		class: className = ''
	}: Props = $props();

	let canvas: HTMLCanvasElement | undefined;
	let ctx: CanvasRenderingContext2D | null = null;
	let rafId = 0;
	let cols = $state(1);
	let rows = $state(1);
	let scale = 1;
	let cssWidth = $state(1);
	let cssHeight = $state(1);
	let ready = false;
	let configuredCellSize = 0;
	let configuredMaxWidth = 0;
	let configuredMaxHeight = 0;
	let size = 1;
	let appliedFrame: Tensor | undefined;
	let appliedFrameIndex = -1;
	let appliedAnimated = false;
	let appliedCellSize = -1;
	let appliedMaxWidth = -1;
	let appliedMaxHeight = -1;
	let appliedDuration = -1;
	let appliedOpacity = -1;
	let appliedTensorCount = -1;

	let visualScale = new Float32Array(1);
	let fromScale = new Float32Array(1);
	let toScale = new Float32Array(1);

	let alpha = new Float32Array(1);
	let fromAlpha = new Float32Array(1);
	let toAlpha = new Float32Array(1);

	let r = new Float32Array(1);
	let g = new Float32Array(1);
	let b = new Float32Array(1);

	let fromR = new Float32Array(1);
	let fromG = new Float32Array(1);
	let fromB = new Float32Array(1);

	let toR = new Float32Array(1);
	let toG = new Float32Array(1);
	let toB = new Float32Array(1);

	let startAt = new Float64Array(1);
	let endAt = new Float64Array(1);
	let active = new Uint8Array(1);
	let nextPresent = new Uint8Array(1);
	let effectType = new Uint8Array(1);

	let hoverVisible = $state(false);
	let hoverCellX = $state(0);
	let hoverCellY = $state(0);
	let hoverValue: number | null = $state(null);
	let popoverX = $state(0);
	let popoverY = $state(0);

	const currentTensor = $derived(tensors[frameIndex] ?? []);
	const hoverIndex = $derived(hoverCellY * cols + hoverCellX);
	const hoverValueText = $derived(formatCellValue(hoverValue));

	function getFrameWidth(frame: Tensor) {
		return Math.max(0, ...frame.map((row) => row.length));
	}

	function clamp(n: number, min: number, max: number) {
		return Math.max(min, Math.min(max, n));
	}

	function hexToRgb(hex: string): Rgb {
		const normalized = hex.replace('#', '');

		return {
			r: parseInt(normalized.slice(0, 2), 16),
			g: parseInt(normalized.slice(2, 4), 16),
			b: parseInt(normalized.slice(4, 6), 16)
		};
	}

	function lerp(a: number, b: number, t: number) {
		return a + (b - a) * t;
	}

	function smoothstep(t: number) {
		return t * t * (3 - 2 * t);
	}

	function easeOutCubic(t: number) {
		return 1 - Math.pow(1 - t, 3);
	}

	function hashCell(index: number) {
		const x = Math.sin((index + 1) * 12.9898) * 43758.5453;

		return x - Math.floor(x);
	}

	function copyGrid<T extends Float32Array | Float64Array | Uint8Array>(
		from: T,
		to: T,
		oldCols: number,
		oldRows: number,
		newCols: number,
		newRows: number
	) {
		const copyCols = Math.min(oldCols, newCols);
		const copyRows = Math.min(oldRows, newRows);

		for (let row = 0; row < copyRows; row++) {
			for (let col = 0; col < copyCols; col++) {
				to[row * newCols + col] = from[row * oldCols + col];
			}
		}
	}

	function valueToRgb(value: number): Rgb {
		const v = clamp(value, 0, 1);
		const scaled = v * (palette.length - 1);
		const index = Math.floor(scaled);
		const nextIndex = Math.min(index + 1, palette.length - 1);
		const localT = smoothstep(scaled - index);
		const from = palette[index];
		const to = palette[nextIndex];

		return {
			r: lerp(from.r, to.r, localT),
			g: lerp(from.g, to.g, localT),
			b: lerp(from.b, to.b, localT)
		};
	}

	function configureCanvas(nextCols: number, nextRows: number, preserve = true) {
		if (!canvas) return;

		const oldCols = cols;
		const oldRows = rows;

		const oldVisualScale = visualScale;
		const oldFromScale = fromScale;
		const oldToScale = toScale;

		const oldAlpha = alpha;
		const oldFromAlpha = fromAlpha;
		const oldToAlpha = toAlpha;

		const oldR = r;
		const oldG = g;
		const oldB = b;

		const oldFromR = fromR;
		const oldFromG = fromG;
		const oldFromB = fromB;

		const oldToR = toR;
		const oldToG = toG;
		const oldToB = toB;

		const oldStartAt = startAt;
		const oldEndAt = endAt;
		const oldActive = active;
		const oldEffectType = effectType;

		cols = Math.max(1, nextCols);
		rows = Math.max(1, nextRows);
		size = cols * rows;
		configuredCellSize = cellSize;
		configuredMaxWidth = maxWidth;
		configuredMaxHeight = maxHeight;

		visualScale = new Float32Array(size);
		fromScale = new Float32Array(size);
		toScale = new Float32Array(size);

		alpha = new Float32Array(size);
		fromAlpha = new Float32Array(size);
		toAlpha = new Float32Array(size);

		r = new Float32Array(size);
		g = new Float32Array(size);
		b = new Float32Array(size);

		fromR = new Float32Array(size);
		fromG = new Float32Array(size);
		fromB = new Float32Array(size);

		toR = new Float32Array(size);
		toG = new Float32Array(size);
		toB = new Float32Array(size);

		startAt = new Float64Array(size);
		endAt = new Float64Array(size);
		active = new Uint8Array(size);
		nextPresent = new Uint8Array(size);
		effectType = new Uint8Array(size);

		if (preserve && oldCols > 0 && oldRows > 0) {
			copyGrid(oldVisualScale, visualScale, oldCols, oldRows, cols, rows);
			copyGrid(oldFromScale, fromScale, oldCols, oldRows, cols, rows);
			copyGrid(oldToScale, toScale, oldCols, oldRows, cols, rows);

			copyGrid(oldAlpha, alpha, oldCols, oldRows, cols, rows);
			copyGrid(oldFromAlpha, fromAlpha, oldCols, oldRows, cols, rows);
			copyGrid(oldToAlpha, toAlpha, oldCols, oldRows, cols, rows);

			copyGrid(oldR, r, oldCols, oldRows, cols, rows);
			copyGrid(oldG, g, oldCols, oldRows, cols, rows);
			copyGrid(oldB, b, oldCols, oldRows, cols, rows);

			copyGrid(oldFromR, fromR, oldCols, oldRows, cols, rows);
			copyGrid(oldFromG, fromG, oldCols, oldRows, cols, rows);
			copyGrid(oldFromB, fromB, oldCols, oldRows, cols, rows);

			copyGrid(oldToR, toR, oldCols, oldRows, cols, rows);
			copyGrid(oldToG, toG, oldCols, oldRows, cols, rows);
			copyGrid(oldToB, toB, oldCols, oldRows, cols, rows);

			copyGrid(oldStartAt, startAt, oldCols, oldRows, cols, rows);
			copyGrid(oldEndAt, endAt, oldCols, oldRows, cols, rows);
			copyGrid(oldActive, active, oldCols, oldRows, cols, rows);
			copyGrid(oldEffectType, effectType, oldCols, oldRows, cols, rows);
		}

		scale = Math.min(1, maxWidth / (cols * cellSize), maxHeight / (rows * cellSize));
		cssWidth = Math.max(1, Math.floor(cols * cellSize * scale));
		cssHeight = Math.max(1, Math.floor(rows * cellSize * scale));

		const dpr = window.devicePixelRatio || 1;

		const nextCanvasWidth = Math.max(1, Math.floor(cssWidth * dpr));
		const nextCanvasHeight = Math.max(1, Math.floor(cssHeight * dpr));

		if (canvas.width !== nextCanvasWidth || canvas.height !== nextCanvasHeight) {
			canvas.width = nextCanvasWidth;
			canvas.height = nextCanvasHeight;
		}

		canvas.style.width = `${cssWidth}px`;
		canvas.style.height = `${cssHeight}px`;

		ctx = canvas.getContext('2d');

		if (!ctx) return;

		ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
		ctx.imageSmoothingEnabled = false;
	}

	function formatCellValue(value: number | null) {
		if (value === null) return 'none';
		if (Number.isInteger(value)) return String(value);

		return String(Number(value.toFixed(6)));
	}

	function getValueAtCell(cellX: number, cellY: number) {
		const value = currentTensor[cellY]?.[cellX];

		if (!Number.isFinite(value)) return null;

		return value;
	}

	function updateCellAt(index: number, now: number) {
		if (!active[index]) return;
		if (now < startAt[index]) return;

		const rawT = clamp((now - startAt[index]) / Math.max(1, endAt[index] - startAt[index]), 0, 1);
		const t = smoothstep(rawT);

		if (effectType[index] === 1) {
			visualScale[index] = lerp(0.12, toScale[index], easeOutCubic(t));
		} else {
			visualScale[index] = lerp(fromScale[index], toScale[index], t);
		}

		alpha[index] = lerp(fromAlpha[index], toAlpha[index], t);
		r[index] = lerp(fromR[index], toR[index], t);
		g[index] = lerp(fromG[index], toG[index], t);
		b[index] = lerp(fromB[index], toB[index], t);

		if (rawT >= 1) {
			visualScale[index] = toScale[index];
			alpha[index] = toAlpha[index];
			r[index] = toR[index];
			g[index] = toG[index];
			b[index] = toB[index];
			active[index] = 0;
			effectType[index] = 0;
		}
	}

	function scheduleCell(
		index: number,
		options: {
			nextScale: number;
			nextAlpha: number;
			nextR: number;
			nextG: number;
			nextB: number;
			delay: number;
			duration?: number;
			type?: 0 | 1 | 2 | 3;
			now?: number;
		}
	) {
		const now = options.now ?? performance.now();

		updateCellAt(index, now);

		fromScale[index] = visualScale[index];
		toScale[index] = options.nextScale;

		fromAlpha[index] = alpha[index];
		toAlpha[index] = options.nextAlpha;

		fromR[index] = r[index];
		fromG[index] = g[index];
		fromB[index] = b[index];

		toR[index] = options.nextR;
		toG[index] = options.nextG;
		toB[index] = options.nextB;

		startAt[index] = now + options.delay;
		endAt[index] = startAt[index] + (options.duration ?? duration);
		active[index] = 1;
		effectType[index] = options.type ?? 3;
	}

	function setCellImmediate(
		index: number,
		options: {
			nextScale: number;
			nextAlpha: number;
			nextR: number;
			nextG: number;
			nextB: number;
		}
	) {
		visualScale[index] = options.nextScale;
		alpha[index] = options.nextAlpha;
		r[index] = options.nextR;
		g[index] = options.nextG;
		b[index] = options.nextB;

		fromScale[index] = options.nextScale;
		toScale[index] = options.nextScale;
		fromAlpha[index] = options.nextAlpha;
		toAlpha[index] = options.nextAlpha;
		fromR[index] = options.nextR;
		fromG[index] = options.nextG;
		fromB[index] = options.nextB;
		toR[index] = options.nextR;
		toG[index] = options.nextG;
		toB[index] = options.nextB;
		active[index] = 0;
		effectType[index] = 0;
	}

	function applyFrame(frame: Tensor, withAnimation = true) {
		if (size === 0) return;

		nextPresent.fill(0);

		const totalCells = Math.max(1, size);
		const enterDuration = Math.min(220, Math.max(90, duration * 0.28));
		const delayWindow = Math.max(0, duration - enterDuration);
		const startedAt = performance.now();

		for (let rowIndex = 0; rowIndex < frame.length; rowIndex++) {
			if (rowIndex >= rows) continue;

			const row = frame[rowIndex];

			for (let columnIndex = 0; columnIndex < row.length; columnIndex++) {
				if (columnIndex >= cols) continue;

				const value = row[columnIndex];

				if (!Number.isFinite(value)) continue;

				const index = rowIndex * cols + columnIndex;
				const color = valueToRgb(value);
				const isEntering = alpha[index] <= 0.001 && toAlpha[index] <= 0.001;
				const delay =
					withAnimation && isEntering
						? (Math.floor(hashCell(index) * totalCells) / Math.max(1, totalCells - 1)) *
							delayWindow
						: 0;

				nextPresent[index] = 1;

				if (!withAnimation) {
					setCellImmediate(index, {
						nextScale: 1,
						nextAlpha: opacity,
						nextR: color.r,
						nextG: color.g,
						nextB: color.b
					});
					continue;
				}

				if (isEntering) {
					visualScale[index] = 0.12;
					alpha[index] = 0;
				}

				scheduleCell(index, {
					nextScale: 1,
					nextAlpha: opacity,
					nextR: color.r,
					nextG: color.g,
					nextB: color.b,
					delay,
					duration: isEntering ? enterDuration : duration,
					type: isEntering ? 1 : 3,
					now: startedAt
				});
			}
		}

		for (let index = 0; index < size; index++) {
			if (nextPresent[index]) continue;

			const isVisibleOrScheduled =
				visualScale[index] > 0.001 || alpha[index] > 0.001 || toAlpha[index] > 0.001 || active[index];

			if (!isVisibleOrScheduled) continue;

			if (!withAnimation) {
				setCellImmediate(index, {
					nextScale: 0,
					nextAlpha: 0,
					nextR: r[index],
					nextG: g[index],
					nextB: b[index]
				});
				continue;
			}

			scheduleCell(index, {
				nextScale: 0,
				nextAlpha: 0,
				nextR: r[index],
				nextG: g[index],
				nextB: b[index],
				delay: 0,
				type: 2,
				now: startedAt
			});
		}

		startRenderLoop();
	}

	function drawFrame() {
		if (!ctx) return false;

		const now = performance.now();
		let hasActiveAnimation = false;

		ctx.clearRect(0, 0, cols * cellSize, rows * cellSize);

		for (let index = 0; index < size; index++) {
			if (active[index]) {
				updateCellAt(index, now);
			}

			if (active[index]) {
				hasActiveAnimation = true;
			}

			const cellScale = visualScale[index];
			const cellAlpha = alpha[index];

			if (cellScale <= 0.001 || cellAlpha <= 0.001) continue;

			const columnIndex = index % cols;
			const rowIndex = Math.floor(index / cols);
			const rectSize = cellSize * cellScale;
			const left = columnIndex * cellSize + (cellSize - rectSize) / 2;
			const top = rowIndex * cellSize + (cellSize - rectSize) / 2;

			ctx.globalAlpha = cellAlpha;
			ctx.fillStyle = `rgb(${clamp(Math.round(r[index]), 0, 255)}, ${clamp(
				Math.round(g[index]),
				0,
				255
			)}, ${clamp(Math.round(b[index]), 0, 255)})`;
			ctx.fillRect(left, top, rectSize, rectSize);
		}

		ctx.globalAlpha = 1;

		return hasActiveAnimation;
	}

	function render() {
		const shouldContinue = drawFrame();

		if (shouldContinue) {
			rafId = requestAnimationFrame(render);
		} else {
			rafId = 0;
		}
	}

	function startRenderLoop() {
		if (rafId) cancelAnimationFrame(rafId);

		render();
	}

	function syncFrame(
		frame: Tensor,
		withAnimation = animated,
		tracked?: {
			frameIndex: number;
			cellSize: number;
			maxWidth: number;
			maxHeight: number;
			duration: number;
			opacity: number;
			tensorCount: number;
		}
	) {
		const nextCols = Math.max(1, getFrameWidth(frame));
		const nextRows = Math.max(1, frame.length);
		const needsConfigure =
			cols !== nextCols ||
			rows !== nextRows ||
			configuredCellSize !== cellSize ||
			configuredMaxWidth !== maxWidth ||
			configuredMaxHeight !== maxHeight;

		if (needsConfigure) {
			configureCanvas(nextCols, nextRows, true);
		}

		applyFrame(frame, withAnimation);
		hidePopover();

		appliedFrame = frame;
		appliedFrameIndex = tracked?.frameIndex ?? frameIndex;
		appliedAnimated = withAnimation;
		appliedCellSize = tracked?.cellSize ?? cellSize;
		appliedMaxWidth = tracked?.maxWidth ?? maxWidth;
		appliedMaxHeight = tracked?.maxHeight ?? maxHeight;
		appliedDuration = tracked?.duration ?? duration;
		appliedOpacity = tracked?.opacity ?? opacity;
		appliedTensorCount = tracked?.tensorCount ?? tensors.length;
	}

	function hidePopover() {
		hoverVisible = false;
		hoverValue = null;
	}

	function handlePointerMove(event: PointerEvent) {
		if (!interactive || !ready || !canvas) {
			hidePopover();
			return;
		}

		const rect = canvas.getBoundingClientRect();
		const localX = event.clientX - rect.left;
		const localY = event.clientY - rect.top;
		const cellX = Math.floor((localX / rect.width) * cols);
		const cellY = Math.floor((localY / rect.height) * rows);

		if (cellX < 0 || cellX >= cols || cellY < 0 || cellY >= rows) {
			hidePopover();
			return;
		}

		hoverVisible = true;
		hoverCellX = cellX;
		hoverCellY = cellY;
		hoverValue = getValueAtCell(cellX, cellY);
		popoverX = localX + 12;
		popoverY = localY + 12;
	}

	onMount(() => {
		ready = true;
	});

	$effect(() => {
		if (!ready) return;

		const frame = currentTensor;
		const shouldAnimate = animated;
		const tracked = {
			frameIndex,
			cellSize,
			maxWidth,
			maxHeight,
			duration,
			opacity,
			tensorCount: tensors.length
		};
		const alreadyApplied =
			frame === appliedFrame &&
			tracked.frameIndex === appliedFrameIndex &&
			shouldAnimate === appliedAnimated &&
			tracked.cellSize === appliedCellSize &&
			tracked.maxWidth === appliedMaxWidth &&
			tracked.maxHeight === appliedMaxHeight &&
			tracked.duration === appliedDuration &&
			tracked.opacity === appliedOpacity &&
			tracked.tensorCount === appliedTensorCount;

		if (alreadyApplied) return;

		untrack(() => {
			syncFrame(frame, shouldAnimate, tracked);
		});
	});

	onDestroy(() => {
		cancelAnimationFrame(rafId);
	});
</script>

<div
	class={`relative inline-block max-w-full touch-none ${className}`}
	style={`width: ${cssWidth}px; height: ${cssHeight}px;`}
	onpointermove={handlePointerMove}
	onpointerleave={hidePopover}
	role={interactive ? 'img' : undefined}
	aria-label={interactive ? `tensor ${cols} by ${rows}` : undefined}
>
	<canvas bind:this={canvas} class="block [image-rendering:pixelated]"></canvas>

	{#if hoverVisible}
		<div
			class="pointer-events-none absolute z-20 min-w-28 rounded-lg border border-zinc-900/10 bg-zinc-900/90 px-2 py-1.5 text-[11px] leading-[1.45] whitespace-nowrap text-white shadow-[0_8px_20px_rgba(0,0,0,0.16)] backdrop-blur-md"
			style={`left: ${popoverX}px; top: ${popoverY}px;`}
		>
			<div>index: {hoverIndex}</div>
			<div>row: {hoverCellY}, col: {hoverCellX}</div>
			<div>value: {hoverValueText}</div>
		</div>
	{/if}
</div>
