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
    // For Ollama, we need to ensure the stream property is properly set
    // Ollama works better with streaming disabled by default
    // But preserve the original request's streaming preference when specified
    const updatedRequest = { ...request };
    
    // Use OLLAMA_MODEL environment variable if set, otherwise use the config model
    const ollamaModel = process.env['OLLAMA_MODEL'];
    if (ollamaModel) {
      updatedRequest.model = ollamaModel;
    } else {
      // Use the default model from the config if no OLLAMA_MODEL is set
      updatedRequest.model = this.contentGeneratorConfig.model || request.model;
    }
    
    if (updatedRequest.stream === undefined) {
      updatedRequest.stream = false; // Default to non-streaming for Ollama when not specified
    }
    // If stream is explicitly set in the original request, preserve it
    
    return updatedRequest;
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