import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { OllamaOpenAICompatibleProvider } from './ollama.js';
import type { Config } from '../../../config/config.js';
import type { ContentGeneratorConfig } from '../../contentGenerator.js';
import type OpenAI from 'openai';

describe('OllamaOpenAICompatibleProvider', () => {
  let mockConfig: ContentGeneratorConfig;
  let mockCliConfig: Config;

  beforeEach(() => {
    mockConfig = {
      apiKey: 'test-key',
      baseUrl: 'http://localhost:11434/v1',
      model: 'qwen:0.5b',
    };
    mockCliConfig = {
      getCliVersion: vi.fn().mockReturnValue('0.0.14'),
    } as unknown as Config;
  });

  afterEach(() => {
    // Clean up environment variables after each test
    delete process.env['OLLAMA_HOST'];
    delete process.env['OLLAMA_MODEL'];
  });

  it('should create an instance', () => {
    const provider = new OllamaOpenAICompatibleProvider(
      mockConfig,
      mockCliConfig
    );
    expect(provider).toBeInstanceOf(OllamaOpenAICompatibleProvider);
  });

  it('should build headers correctly', () => {
    const provider = new OllamaOpenAICompatibleProvider(
      mockConfig,
      mockCliConfig
    );
    const headers = provider.buildHeaders();
    expect(headers['User-Agent']).toContain('QwenCode/0.0.14');
  });

  it('should build client with default Ollama URL when no baseUrl provided', () => {
    delete mockConfig.baseUrl; // Remove baseUrl to test default
    const provider = new OllamaOpenAICompatibleProvider(
      mockConfig,
      mockCliConfig
    );
    const client = provider.buildClient();
    // The client's baseURL should be the default Ollama URL
    expect(client.baseURL).toContain('11434'); // Check that it includes the default port
  });

  it('should build client with custom base URL when provided', () => {
    mockConfig.baseUrl = 'http://my-ollama.local:11434/v1';
    const provider = new OllamaOpenAICompatibleProvider(
      mockConfig,
      mockCliConfig
    );
    const client = provider.buildClient();
    expect(client.baseURL).toBe('http://my-ollama.local:11434/v1');
  });

  it('should build request with OLLAMA_MODEL environment variable', () => {
    process.env['OLLAMA_MODEL'] = 'qwen2:7b';
    const provider = new OllamaOpenAICompatibleProvider(
      mockConfig,
      mockCliConfig
    );
    const originalRequest: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'Hello' }],
      temperature: 0.7,
      stream: false,
    };
    const modifiedRequest = provider.buildRequest(originalRequest, 'test-prompt-id');
    expect(modifiedRequest.model).toBe('qwen2:7b');
  });

  it('should use default model when OLLAMA_MODEL is not set', () => {
    const provider = new OllamaOpenAICompatibleProvider(
      mockConfig,
      mockCliConfig
    );
    const originalRequest: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'Hello' }],
      temperature: 0.7,
      stream: false,
    };
    const modifiedRequest = provider.buildRequest(originalRequest, 'test-prompt-id');
    expect(modifiedRequest.model).toBe('qwen:0.5b'); // Default model
  });

  it('should detect Ollama provider when OLLAMA_HOST is set', () => {
    process.env['OLLAMA_HOST'] = 'http://localhost:11434';
    const result = OllamaOpenAICompatibleProvider.isOllamaProvider(mockConfig);
    expect(result).toBe(true);
  });

  it('should detect Ollama provider when OLLAMA_MODEL is set', () => {
    process.env['OLLAMA_MODEL'] = 'qwen:0.5b';
    const result = OllamaOpenAICompatibleProvider.isOllamaProvider(mockConfig);
    expect(result).toBe(true);
  });

  it('should detect Ollama provider when baseUrl contains 11434', () => {
    mockConfig.baseUrl = 'http://localhost:11434/v1';
    const result = OllamaOpenAICompatibleProvider.isOllamaProvider(mockConfig);
    expect(result).toBe(true);
  });

  it('should not detect Ollama provider when no indicators are present', () => {
    delete process.env['OLLAMA_HOST'];
    delete process.env['OLLAMA_MODEL'];
    mockConfig.baseUrl = 'https://api.openai.com/v1';
    const result = OllamaOpenAICompatibleProvider.isOllamaProvider(mockConfig);
    expect(result).toBe(false);
  });
});