export class PtrClient {
  constructor({ baseUrl, fetch: fetchImpl = globalThis.fetch }) {
    if (!baseUrl) throw new TypeError("baseUrl is required");
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.fetch = fetchImpl;
  }

  async health() {
    return this.#json("/health", { method: "GET" });
  }

  async request(input, { signal } = {}) {
    if (!input?.id || typeof input.text !== "string") {
      throw new TypeError("request requires id and text");
    }
    return this.#json("/v1/requests", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
      signal,
    });
  }

  async #json(path, init) {
    const response = await this.fetch(this.baseUrl + path, init);
    if (!response.ok) {
      const text = await response.text();
      throw new PtrHttpError(response.status, text);
    }
    return response.json();
  }
}

export class PtrHttpError extends Error {
  constructor(status, body) {
    super(`PTR HTTP ${status}: ${body}`);
    this.name = "PtrHttpError";
    this.status = status;
    this.body = body;
  }
}
