export interface PtrClientOptions {
  baseUrl: string;
  fetch?: typeof globalThis.fetch;
}

export interface PtrRequest {
  id: string;
  text: string;
}

export interface PtrResponse {
  id: string;
  revision: number;
  text: string;
}

export interface PtrHealth {
  status: string;
  version?: string;
}

export class PtrHttpError extends Error {
  readonly status: number;
  readonly body: string;
  constructor(status: number, body: string);
}

export class PtrClient {
  constructor(options: PtrClientOptions);
  health(): Promise<PtrHealth>;
  request(input: PtrRequest, options?: { signal?: AbortSignal }): Promise<PtrResponse>;
}
