import OpenAI from 'openai';
import type { Config } from '../../../config/config.js';
import type { ContentGeneratorConfig } from '../../contentGenerator.js';
import { DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES } from '../constants.js';
import type { OpenAICompatibleProvider } from './types.js';

/**
 * Ollama provider for OpenAI-compatible API
 */
export class OllamaOpenAICompatibleProvider
  implements OpenAICompatibleProvider
{
  protected contentGeneratorConfig: ContentGeneratorConfig;
  protected cliConfig: Config;

  constructor(
    contentGeneratorConfig: ContentGeneratorConfig,
    cliConfig: Config,
  ) {
    this.cliConfig = cliConfig;
    this.contentGeneratorConfig = contentGeneratorConfig;
  }

  buildHeaders(): Record<string, string | undefined> {
    const version = this.cliConfig.getCliVersion() || 'unknown';
    const userAgent = `QwenCode/${version} (${process.platform}; ${process.arch})`;
    return {
      'User-Agent': userAgent,
    };
  }

  buildClient(): OpenAI {
    const {
      apiKey,
      baseUrl,
      timeout = DEFAULT_TIMEOUT,
      maxRetries = DEFAULT_MAX_RETRIES,
    } = this.contentGeneratorConfig;
    
    const defaultHeaders = this.buildHeaders();
    
    // For Ollama, we typically don't need an API key, but if provided, use it
    const clientApiKey = process.env['OLLAMA_API_KEY'] || apiKey || 'ollama'; // Default key for Ollama if needed
    
    // Use OLLAMA_HOST if set, otherwise use the provided baseUrl or default
    const ollamaBaseUrl = process.env['OLLAMA_HOST'] 
      ? `${process.env['OLLAMA_HOST']}/v1`
      : baseUrl || 'http://localhost:11434/v1';
    
    return new OpenAI({
      apiKey: clientApiKey,
      baseURL: ollamaBaseUrl,
      timeout,
      maxRetries,
      defaultHeaders,
    });
  }

  buildRequest(
    request: OpenAI.Chat.ChatCompletionCreateParams,
    _userPromptId: string,
  ): OpenAI.Chat.ChatCompletionCreateParams {
    // For Ollama, ensure the model specified matches the OLLAMA_MODEL environment variable
    const ollamaModel = process.env['OLLAMA_MODEL'] || 'qwen:0.5b';
    
    // For Ollama, we need to ensure the stream property is properly set
    // Ollama works better with streaming disabled by default
    if ('stream' in request) {
      return {
        ...request,
        model: ollamaModel, // Override the model with the Ollama model
        stream: request.stream ?? false, // Ensure stream property is explicitly set
      } as OpenAI.Chat.ChatCompletionCreateParams;
    }
    
    // If stream is not in the request, add it explicitly
    return {
      ...request,
      model: ollamaModel, // Override the model with the Ollama model
      stream: false, // Default to non-streaming for Ollama
    } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming;
  }

  static isOllamaProvider(
    contentGeneratorConfig: ContentGeneratorConfig,
  ): boolean {
    // Check if OLLAMA_HOST or OLLAMA_MODEL environment variables are set
    const hasOllamaHost = !!process.env['OLLAMA_HOST'];
    const hasOllamaModel = !!process.env['OLLAMA_MODEL'];
    
    // Also check if the base URL matches common Ollama patterns
    const hasOllamaBaseUrl = !!contentGeneratorConfig?.baseUrl?.includes('11434');
    
    return hasOllamaHost || hasOllamaModel || hasOllamaBaseUrl;
  }
}