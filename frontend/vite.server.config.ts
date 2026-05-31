import { defineConfig } from 'vite';
import { builtinModules } from 'node:module';

export default defineConfig({
	build: {
		target: 'node20',
		outDir: 'build',
		emptyOutDir: false,
		rollupOptions: {
			input: {
				server: './src/server.ts'
			},
			external: [...builtinModules, ...builtinModules.map((m) => `node:${m}`), './handler.js'],
			output: {
				entryFileNames: 'server.js',
				format: 'es'
			}
		}
	}
});
