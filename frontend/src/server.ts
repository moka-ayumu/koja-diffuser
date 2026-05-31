import { Hono } from 'hono';
import { proxy } from 'hono/proxy';
import { serve } from '@hono/node-server';
import type { HttpBindings } from '@hono/node-server';
import { RESPONSE_ALREADY_SENT } from '@hono/node-server/utils/response';

import { handler as svelteKitHandler } from './handler.js';

const app = new Hono<{ Bindings: HttpBindings }>();

const API_ORIGIN = 'http://localhost:8000';

app.all('/api/*', async (c) => {
	const incomingUrl = new URL(c.req.url);

	const pathWithoutApi = incomingUrl.pathname.replace(/^\/api/, '') || '/';

	const targetUrl = new URL(pathWithoutApi + incomingUrl.search, API_ORIGIN);

	return proxy(targetUrl.toString(), {
		...c.req,
		headers: {
			...Object.fromEntries(c.req.raw.headers),
			host: new URL(API_ORIGIN).host
		}
	});
});

app.use('*', async (c) => {
	const req = c.env.incoming;
	const res = c.env.outgoing;

	await new Promise<void>((resolve, reject) => {
		let settled = false;

		const done = () => {
			if (settled) return;
			settled = true;
			cleanup();
			resolve();
		};

		const fail = (err: unknown) => {
			if (settled) return;
			settled = true;
			cleanup();
			reject(err);
		};

		const cleanup = () => {
			res.off('finish', done);
			res.off('close', done);
			res.off('error', fail);
		};

		res.once('finish', done);
		res.once('close', done);
		res.once('error', fail);

		try {
			svelteKitHandler(req, res, (err?: unknown) => {
				if (err) fail(err);
				else done();
			});
		} catch (err) {
			fail(err);
		}
	});

	return RESPONSE_ALREADY_SENT;
});

serve({
	fetch: app.fetch,
	port: 7860,
	hostname: '0.0.0.0'
});
