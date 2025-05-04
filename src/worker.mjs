--- START OF FILE worker.mjs ---

import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      // Check if err is an HttpError and return its status, otherwise use 500
      const status = (err instanceof HttpError) ? err.status : (err.status ?? 500);
      return new Response(err.message, fixCors({ status }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success, message, status = 400) => {
        if (!success) {
          throw new HttpError(message, status);
        }
      };
      const { pathname } = new URL(request.url);

      // Normalize pathname by removing potential leading /v1 if present
      // Some clients might send /v1/chat/completions, others just /chat/completions
      // Let's keep endsWith for flexibility with base URLs
      // const normalizedPathname = pathname.startsWith('/v1') ? pathname.substring(3) : pathname;

      switch (true) {
        // Handles chat models (including multimodal like image generation through chat)
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST", "Method Not Allowed", 405);
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);

        // Handles embeddings models
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST", "Method Not Allowed", 405);
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);

        // Handles listing models
        case pathname.endsWith("/models"):
          assert(request.method === "GET", "Method Not Allowed", 405);
          return handleModels(apiKey)
            .catch(errHandler);

        // --- NEW CASE FOR IMAGE GENERATION ---
        // Handles OpenAI-style image generation endpoint
        case pathname.endsWith("/images/generations"):
          assert(request.method === "POST", "Method Not Allowed", 405);
          return handleImageGenerations(await request.json(), apiKey)
             .catch(errHandler);
        // --- END NEW CASE ---

        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  // Add other common headers if needed
   headers.set("Access-Control-Allow-Methods", "*");
   headers.set("Access-Control-Allow-Headers", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*", // Allow all methods for simplicity
      "Access-Control-Allow-Headers": "*", // Allow all headers for simplicity
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta"; // Use v1beta for broad model support

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // Use a recent version or the one you prefer
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels (apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let body;
  const responseClone = response.clone();
  let responseBodyText;

  try {
    responseBodyText = await responseClone.text();
  } catch (e) {
    console.error("Error reading response text:", e);
    return new Response(response.body, fixCors(response));
  }

  if (response.ok) {
    try {
      const { models } = JSON.parse(responseBodyText);
      body = JSON.stringify({
        object: "list",
        data: models.map(({ name }) => ({
          id: name.replace("models/", ""),
          object: "model",
          created: 0,
          owned_by: "google",
        })),
      }, null, "  ");
    } catch (err) {
      console.error("Error processing models response:", err);
      return new Response(responseBodyText, fixCors(response));
    }
  } else {
    body = responseBodyText;
  }

  return new Response(body, fixCors(response));
}


const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKey) {
  // req is the parsed JSON request body from the client

  let modelName = req.model; // Get the model name from the request

  if (typeof modelName !== "string" || !modelName) {
    // If model is not specified or not a string, use the default.
    // In a strict API proxy, one might throw an error instead.
     console.warn(`Model not specified or invalid for embeddings. Using default: ${DEFAULT_EMBEDDINGS_MODEL}`);
     modelName = DEFAULT_EMBEDDINGS_MODEL;
  }

   // Ensure the model name has the 'models/' prefix for the Google API endpoint
  const modelEndpoint = modelName.startsWith("models/") ? modelName : "models/" + modelName;

  // Input can be a string or array of strings
  if (!Array.isArray(req.input)) {
    // If input is a single value (string, number, etc.), wrap it in an array.
    // Ensure it's treated as text input.
     if (req.input === null || req.input === undefined) {
         throw new HttpError("Input is required for embeddings.", 400);
     }
    req.input = [ String(req.input) ]; // Ensure array of strings
  } else {
     // Ensure all items in the input array are strings
     req.input = req.input.map(item => String(item));
  }


  // API endpoint is batchEmbedContents for multiple inputs
  // Or embedContent for single input. batchEmbedContents is more flexible.
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${modelEndpoint}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        // model needs to be included in each request for batch endpoint (redundant but required by API)
        model: modelEndpoint,
        content: { parts: [{ text }] }, // Input is text parts
        outputDimensionality: req.dimensions, // Optional: specify dimension
        // Add other potential embedding request parameters if needed
      }))
    })
  });

  let body;
  const responseClone = response.clone();
  let responseBodyText;
    try {
      responseBodyText = await responseClone.text();
    } catch (e) {
       console.error("Error reading response text:", e);
       return new Response(response.body, fixCors(response));
    }


  if (response.ok) {
    try {
      const data = JSON.parse(responseBodyText);
       if (!data.embeddings || !Array.isArray(data.embeddings)) {
           // Even on success, API might return empty embeddings or different structure
           // Check if it's a valid embedding response format
           if (data.error) {
               // API returned an error structure despite 200 status? Rare but possible.
                console.error("Google API error in embeddings JSON body:", data.error);
                // Return the original error JSON with potentially correct status if available
                const errorStatus = data.error.code && typeof data.error.code === 'number' ? data.error.code : response.status;
                return new Response(responseBodyText, fixCors({ status: errorStatus, headers: response.headers }));
           }
           throw new Error("Unexpected response format: 'embeddings' array not found or invalid.");
       }

      body = JSON.stringify({
        object: "list", // OpenAI standard list object for embeddings
        data: data.embeddings.map(({ values }, index) => ({
          object: "embedding",
          index, // Original index from the input request order
          embedding: values,
        })),
        model: modelName, // Report the requested model name (without 'models/' prefix)
        // Add usage if available from the API response
         usage: data.usageMetadata ? transformUsage(data.usageMetadata) : undefined,
      }, null, "  "); // Pretty print for debugging
    } catch (err) {
        console.error("Error processing embeddings response:", err);
        // If transformation fails, return the original (potentially error) response body text
        return new Response(responseBodyText, fixCors(response));
    }
  } else {
     // If the original response was not ok, return its body (likely an API error message)
     body = responseBodyText;
     // No need to process JSON if status is not ok, just return the error body.
  }
  return new Response(body, fixCors(response));
}


const DEFAULT_MODEL = "gemini-2.0-flash"; // Default chat model
async function handleCompletions (req, apiKey) {
  let modelName = DEFAULT_MODEL;
  // Determine the model name without the 'models/' prefix for the API call URL
  // and for reporting in the response object.
  if (typeof req.model === "string" && req.model) { // Check if model is a non-empty string
      if (req.model.startsWith("models/")) {
          modelName = req.model.substring(7); // Remove 'models/' prefix
      } else {
          // Assume model name is directly provided (e.g., "gemini-pro", "gemini-2.0-flash-exp-image-generation")
          modelName = req.model;
      }
  }
  // The actual API call URL will be BASE_URL/API_VERSION/models/{model_name}:TASK
  // where model_name is just "gemini-pro", "gemini-2.0-flash", etc.

  let body = await transformRequest(req); // Transforms OpenAI -> Gemini request body

  // Handle specific model suffix requests like "-search-preview" or ":search"
  // This modifies the request body to include Google Search tool.
  // Note: This is specific to certain Gemini models/features.
  const originalModelReq = req.model?.toLowerCase() || "";
  const isSearchModel = originalModelReq.endsWith(":search") || originalModelReq.endsWith("-search-preview");

  if (isSearchModel) {
      // Ensure the modelName used for the API call does *not* include the search suffix
      if (modelName.endsWith(":search")) {
          modelName = modelName.substring(0, modelName.length - 7);
      } else if (modelName.endsWith("-search-preview")) {
          modelName = modelName.substring(0, modelName.length - 16);
      }
      // Add the Google Search tool to the request body
      body.tools = body.tools || [];
      // Ensure the tool is only added once
      if (!body.tools.some(tool => tool.googleSearch)) {
         body.tools.push({googleSearch: {}});
      }
  }

  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${modelName}:${TASK}`;
  if (req.stream) {
      url += "?alt=sse"; // Append alt=sse for Server-Sent Events streaming format
  }

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  let responseBody; // This will hold the final body for the Response object

  if (response.ok) {
    let id = "chatcmpl-" + generateId(); // Generate an OpenAI-like completion ID
    const shared = {}; // Shared state for streaming transforms

    if (req.stream) {
      // Streaming response requires piping through transformations
      responseBody = response.body
        .pipeThrough(new TextDecoderStream()) // Decode bytes to text
        .pipeThrough(new TransformStream({ // Parse SSE "data: " lines
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "", // Buffer for incomplete lines
          shared, // Pass shared state
        }))
        .pipeThrough(new TransformStream({ // Transform Google format chunks to OpenAI stream format
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: req.stream_options?.include_usage, // Option to include usage in stream
          model: req.model, // Pass original model name from request (can include 'models/')
          id, // Pass the generated ID
          last: [], // State for accumulating/tracking last chunk info per candidate
          shared, // Pass shared state
        }))
        .pipeThrough(new TextEncoderStream()); // Encode text back to bytes for response body
    } else {
      // Non-streaming response
      let responseBodyText;
      try {
         const responseClone = response.clone();
         responseBodyText = await responseClone.text();
      } catch (e) {
         console.error("Error reading non-streaming response text:", e);
         // If we can't read as text, try returning the original body stream as is
         return new Response(response.body, fixCors(response));
      }


      try {
        const data = JSON.parse(responseBodyText);
        // Check for expected structure or API errors embedded in JSON
        if (data.error) {
           // Google API often returns { error: { code, message, status }} for non-200 JSON
           console.error("Google API error in JSON body:", data.error);
           // Return the original error JSON with potentially correct status if available
           const errorStatus = data.error.code && typeof data.error.code === 'number' ? data.error.code : response.status;
           return new Response(responseBodyText, fixCors({ status: errorStatus, headers: response.headers }));

        } else if (!data.candidates && !data.promptFeedback) {
          // Successful response should have candidates array (even if empty) or promptFeedback
          // If neither is present, it's an unexpected format
          throw new Error("Invalid completion object: missing 'candidates' or 'promptFeedback'.");
        } else {
           // Transform the Google response object to OpenAI format JSON string
           responseBody = processCompletionsResponse(data, req.model, id);
        }
      } catch (err) {
        console.error("Error parsing or processing non-streaming response:", err);
        // If parsing or transformation fails, return the original response body text
        responseBody = responseBodyText; // Return the raw text body
      }
    }
  } else {
    // If the initial fetch response was not ok (e.g., 401, 404, 500 status)
    // Return the original response body directly, which often contains the error details from Google
    responseBody = response.body;
    // No need to process JSON if status is not ok, just return the error body.
  }

  // Return the final Response object with the appropriate body and CORS headers.
  // fixCors ensures CORS headers are present regardless of success/failure status.
  return new Response(responseBody, fixCors(response));
}


// --- NEW FUNCTION FOR IMAGE GENERATION ---

async function handleImageGenerations(req, apiKey) {
    // This function handles requests to the /images/generations endpoint
    // It expects an OpenAI-like request body for image generation:
    // { prompt: string, n?: number, size?: string, response_format?: { type: "url" | "b64_json" } }
    // It will call the Gemini generateContent API with an image generation model.

    const prompt = req.prompt;
    if (typeof prompt !== 'string' || !prompt) {
        throw new HttpError("The 'prompt' field is required and must be a non-empty string.", 400);
    }

    // Gemini image generation uses a specific model via the generateContent endpoint
    const geminiImageModel = "gemini-2.0-flash-exp-image-generation"; // Use the known image model name

    // Construct the Gemini generateContent request body
    const geminiRequestBody = {
        contents: [{
            role: "user",
            parts: [{ text: prompt }] // Send the prompt as text content
        }],
        // Add safety settings if desired (already defaulted globally, but can override per request)
        safetySettings: safetySettings,
        // Add generation config - Gemini generateContent supports candidateCount (n)
        generationConfig: {
           candidateCount: req.n !== undefined && typeof req.n === 'number' && req.n >= 1 ? Math.min(req.n, 4) : 1, // Limit n to reasonable number, default 1
           // Gemini generateContent for image models doesn't have a 'size' parameter.
           // The model generates images at its native resolution. Ignore req.size.
        }
        // Tools/tool_config are not applicable for simple image generation via this model
    };

     console.log("Calling Gemini Image Model:", geminiImageModel, "with prompt:", prompt);

    // Call the Gemini generateContent API
    const url = `${BASE_URL}/${API_VERSION}/models/${geminiImageModel}:generateContent`;
    const response = await fetch(url, {
        method: "POST",
        headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
        body: JSON.stringify(geminiRequestBody),
    });

    let responseBody;
    const responseClone = response.clone();
    let responseBodyText;
    try {
       responseBodyText = await responseClone.text();
    } catch (e) {
        console.error("Error reading image generation response text:", e);
        return new Response(response.body, fixCors(response));
    }


    if (response.ok) {
        try {
            const data = JSON.parse(responseBodyText);

            // Check for Google API errors embedded in JSON
            if (data.error) {
               console.error("Google API error in image generation JSON body:", data.error);
                const errorStatus = data.error.code && typeof data.error.code === 'number' ? data.error.code : response.status;
                return new Response(responseBodyText, fixCors({ status: errorStatus, headers: response.headers }));

            } else if (!data.candidates && !data.promptFeedback) {
                // If no candidates and no prompt feedback, it's an unexpected successful response
                 throw new Error("Invalid image generation response: missing 'candidates' or 'promptFeedback'.");
            }

             // Handle prompt blocking first
            const choices = [];
            if (checkPromptBlock(choices, data.promptFeedback, "message")) {
                 // If prompt was blocked, checkPromptBlock added a choice with finish_reason: "content_filter"
                 // We need to format this as an OpenAI image generation error response
                 // OpenAI image generation errors are typically not in the `choices` format, but a top-level error or specific structure.
                 // Let's return a structured error response.
                 let errorMessage = data.promptFeedback.blockReason ? `Prompt blocked: ${data.promptFeedback.blockReason}` : "Prompt blocked.";
                 if (data.promptFeedback.safetyRatings) {
                     errorMessage += " Details: " + data.promptFeedback.safetyRatings
                         .filter(r => r.blocked)
                         .map(r => `${r.category} (${r.probability})`)
                         .join(", ");
                 }
                 throw new HttpError(errorMessage, 400); // Return a 400 error for client

                 // Alternatively, try to mimic an OpenAI-like error response body structure if needed
                 // return new Response(JSON.stringify({ error: { message: errorMessage, type: "content_filter", code: "prompt_blocked" } }), fixCors({ status: 400 }));
            }

            // If not blocked, process candidates (should contain inlineData parts for images)
            const imageUrls = [];
            if (data.candidates && Array.isArray(data.candidates)) {
                for (const candidate of data.candidates) {
                    if (candidate.content?.parts && Array.isArray(candidate.content.parts)) {
                        for (const part of candidate.content.parts) {
                            if (part.inlineData && part.inlineData.mimeType && part.inlineData.data) {
                                // Found an inline image part
                                const mimeType = part.inlineData.mimeType;
                                const base64Data = part.inlineData.data;

                                const item = {};
                                // Respect the requested response_format if specified
                                const responseFormatType = req.response_format?.type || 'url'; // Default to 'url'

                                if (responseFormatType === 'b64_json') {
                                    item.b64_json = base64Data;
                                } else { // Default or 'url'
                                    // Create a data URL
                                    item.url = `data:${mimeType};base64,${base64Data}`;
                                }
                                // Add revised_prompt? Gemini doesn't typically provide one for image gen
                                // item.revised_prompt = ...;
                                imageUrls.push(item);
                            }
                             // Ignore non-image parts (like text or function calls) in image generation response
                        }
                    }
                }
            }

            if (imageUrls.length === 0) {
                // If response was OK but no images were found in parts
                 console.warn("Google API returned OK for image generation but found no image parts in candidates:", data);
                 // It might be an empty generation or another issue.
                 // Return an error or an empty data array? Empty array is more standard for OpenAI success.
                 // Let's return an empty array if no images were found but the API call was technically OK.
            }


            // Construct OpenAI-style image generation response
            const openaiResponse = {
                created: Math.floor(Date.now() / 1000), // Unix timestamp
                data: imageUrls, // Array of { url: "..." } or { b64_json: "..." }
                // Add usage info if available (Gemini generateContent might provide usageMetadata)
                usage: data.usageMetadata ? transformUsage(data.usageMetadata) : undefined,
            };

             responseBody = JSON.stringify(openaiResponse); // Compact JSON output

        } catch (err) {
            console.error("Error parsing or processing image generation response:", err);
            // If parsing or transformation fails, return the original response body text
            responseBody = responseBodyText; // Return the raw text body
        }
    } else {
       // If the initial fetch response was not ok (e.g., 401, 404, 500 status from Google)
       // Return the original response body directly
       responseBody = response.body;
    }

    // Return the final Response object with appropriate headers
    return new Response(responseBody, fixCors(response));
}

// --- END NEW FUNCTION ---


// Helper to adjust schema properties for compatibility (e.g., remove additionalProperties: false)
const adjustProps = (schemaPart) => {
  if (typeof schemaPart !== "object" || schemaPart === null) {
    return;
  }
  if (Array.isArray(schemaPart)) {
    schemaPart.forEach(adjustProps);
  } else {
    // If it's an object with properties and specifically set additionalProperties to false, remove it.
    if (schemaPart.type === "object" && schemaPart.properties && schemaPart.additionalProperties === false) {
      delete schemaPart.additionalProperties;
    }
    // Recursively process all values (properties in an object, items in an array)
    Object.values(schemaPart).forEach(adjustProps);
  }
};
// Helper to adjust the overall function schema structure within a tool definition
const adjustSchema = (tool) => {
  if (!tool || tool.type !== "function" || !tool.function || !tool.function.parameters) {
      return; // Not a function tool with parameters
  }
  const parameters = tool.function.parameters;
  if (parameters.type === "object" && parameters.strict !== undefined) {
     delete parameters.strict; // Remove potential 'strict' flag
  }
  adjustProps(parameters); // Adjust properties within the parameters schema
};

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE", // Defaulting to BLOCK_NONE
}));

// Mapping OpenAI request fields to Gemini generationConfig fields
const fieldsMap = {
  frequency_penalty: "frequencyPenalty",
  max_tokens: "maxOutputTokens",
  presence_penalty: "presencePenalty",
  seed: "seed",
  top_k: "topK",
  top_p: "topP",
  // n (candidateCount) is handled separately based on streaming
  // stop (stopSequences) is handled separately
  // response_format is handled separately
};

const transformConfig = (req) => {
  let cfg = {};
  for (const openAiKey in fieldsMap) {
    const geminiKey = fieldsMap[openAiKey];
    if (req[openAiKey] !== undefined) {
      cfg[geminiKey] = req[openAiKey];
    }
  }

  if (req.stop !== undefined) {
      cfg.stopSequences = Array.isArray(req.stop) ? req.stop : [req.stop];
  }
  // Handle n (candidateCount): Only apply if not streaming, as streaming only supports 1.
  if (req.n !== undefined && !req.stream) {
      // Ensure n is a number and at least 1
      if (typeof req.n === 'number' && req.n >= 1) {
          cfg.candidateCount = req.n;
      } else {
          console.warn(`Invalid value for 'n' in request: ${req.n}. Ignoring.`);
      }
  }


  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        if (!req.response_format.json_schema?.schema) {
             throw new HttpError("json_schema response_format requires a 'schema' object.", 400);
        }
        adjustProps(req.response_format.json_schema.schema);
        cfg.responseSchema = req.response_format.json_schema.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
        // eslint-disable-next-line no-fallthrough
      case "json_object":
        cfg.responseMimeType = "application/json";
        // If json_object, ideally provide a minimal schema if none exists and model requires it.
        // For now, just setting mime type.
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError(`Unsupported response_format.type: "${req.response_format.type}"`, 400);
    }
  }
  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch image from URL: ${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      if (!mimeType || !mimeType.startsWith('image/')) {
           console.warn(`Warning: Fetched URL (${url}) did not return an image Content-Type (${mimeType || 'none'}). Attempting to process.`);
           // Try to guess based on extension or default? Or require? Let's allow it but warn.
           // If mimeType is null, default to a generic binary type.
           if (!mimeType) mimeType = 'application/octet-stream';
      }
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new HttpError("Error fetching image from URL: " + err.message, 400);
    }
  } else if (url.startsWith("data:")) {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match || !match.groups?.mimeType || !match.groups?.data) {
      throw new HttpError("Invalid image data URL format.", 400);
    }
    ({ mimeType, data } = match.groups);
     if (!mimeType.startsWith('image/')) {
         console.warn(`Warning: Data URL does not specify an image MIME type (${mimeType}). Attempting to process.`);
     }
  } else {
    throw new HttpError("Unsupported image URL format. Must be http, https, or data URL.", 400);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};


// Transforms OpenAI assistant 'tool_calls' into Gemini 'functionCall' parts.
const transformFnCalls = ({ tool_calls }) => {
  const callsMapping = {}; // Map tool_call_id -> function.name
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type }, i) => {
    if (type !== "function") {
      throw new HttpError(`Unsupported tool_call type in assistant message history: "${type}". Only "function" is supported.`, 400);
    }
    let args;
    try {
      args = JSON.parse(argstr);
    } catch (err) {
      console.error("Error parsing function arguments in assistant message:", err);
      throw new HttpError("Invalid function arguments in assistant message history: " + argstr, 400);
    }
    callsMapping[id] = name; // Store the mapping for potential subsequent 'tool' messages
    return {
      functionCall: {
        // Google API functionCall part uses 'name' and 'args'. 'id' is not typically included here.
        name,
        args,
      }
    };
  });
   // Attach the mapping to the parts array for potential use by subsequent 'tool' messages
  parts.callsMapping = callsMapping;
  return parts;
};

// Transforms OpenAI `messages` array into Gemini `contents` array and `system_instruction`.
const transformMessages = async (messages) => {
  if (!messages || !Array.isArray(messages)) {
      if (messages === undefined) return { contents: [] };
      throw new HttpError("messages must be an array.", 400);
  }

  const contents = [];
  let system_instruction_parts = []; // Accumulate system parts
  let lastAssistantToolCallsMapping = {}; // Store mapping from the previous assistant message tool_calls

  for (const item of messages) {
    const message = JSON.parse(JSON.stringify(item)); // Deep clone message

    switch (message.role) {
      case "system":
         // system message content is always string in OpenAI chat completions
         if (typeof message.content === 'string' && message.content.trim().length > 0) {
             // Add text part(s) from system message to system_instruction accumulator
             system_instruction_parts.push({ text: message.content });
         } else {
             console.warn("Skipping empty or invalid system message content.");
         }
        continue; // Skip adding system message to contents array

      case "tool": {
        // Tool message is the result of a function call from the assistant.
        // It corresponds to a 'functionResponse' part in the *user's* content turn.
        if (!message.tool_call_id) {
             throw new HttpError("tool_call_id is required for messages with role 'tool'.", 400);
        }
        // Need the function name associated with this tool_call_id from the *preceding assistant message*.
        const functionName = lastAssistantToolCallsMapping[message.tool_call_id];
        if (!functionName) {
             // This is a strict requirement for mapping to Gemini functionResponse
             throw new HttpError(`Function name not found for tool_call_id "${message.tool_call_id}". Ensure the immediately preceding assistant message contained this tool_call_id.`, 400);
        }

        let responseData;
        try {
            // Content of a tool message is the stringified JSON result.
            responseData = JSON.parse(message.content);
        } catch (err) {
             console.error("Error parsing tool message content:", err);
             throw new HttpError("Invalid content in tool message (not valid JSON).", 400);
        }

        // The 'tool' message itself *is* the user's turn responding to the call.
        // Create a new 'user' content entry specifically for this function response.
        // This matches Google API examples where user turn can have functionResponse parts.
        // Note: OpenAI 'tool' messages only have role, tool_call_id, content. No other parts.
        contents.push({
            role: "user",
            parts: [{
                 functionResponse: {
                    name: functionName,
                    response: responseData,
                 }
            }]
        });

        lastAssistantToolCallsMapping = {}; // Reset mapping after a user turn
        break; // Processed tool message, continue to next item

      } // end case "tool"


      case "assistant":
        message.role = "model"; // Transform role for Google API

        if (message.tool_calls && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
             // Transform OpenAI tool_calls into Gemini functionCall parts
             const toolCallParts = transformFnCalls(message);
             message.parts = toolCallParts;
             // Store the mapping from this assistant message's tool calls for the *next* tool message
             lastAssistantToolCallsMapping = toolCallParts.callsMapping || {};

             // OpenAI spec: 'content' is required unless tool_calls is specified.
             // If tool_calls are present, 'content' should typically be null or empty.
             // If it exists and is not empty, decide whether to include it.
             if (message.content !== undefined && message.content !== null && message.content !== "") {
                 console.warn("Assistant message has both tool_calls and non-empty content. Prioritizing tool_calls and ignoring content.");
                 // If you needed to include the text content alongside function calls,
                 // you'd need to process message.content (string or array) here
                 // and add its parts to message.parts array.
             }
             delete message.content; // Remove the OpenAI 'content' field

        } else {
             // Process standard text/image content for assistant message
             message.parts = await transformMsgContent(message); // Handles array or string content
             lastAssistantToolCallsMapping = {}; // Reset mapping as this message has no tool calls
             delete message.content; // Remove the OpenAI 'content' field
        }
         // Add the transformed message to contents
        contents.push({
            role: message.role,
            parts: message.parts
        });
        break; // Done with assistant message

      case "user":
        message.role = "user"; // Role is already correct

        // Process content for user message (handles array of text/image/audio)
        message.parts = await transformMsgContent(message);

         // If the user message content was empty after transformation (e.g., empty string or empty array),
         // Google API might require a minimal part. Add an empty text part.
         if (!message.parts || message.parts.length === 0) {
             message.parts = [{ text: "" }]; // Add a minimal text part
             console.warn("Empty user message content received, adding empty text part.");
         }

         delete message.content; // Remove the OpenAI 'content' field
        lastAssistantToolCallsMapping = {}; // Reset mapping after a user turn

        // Add the transformed message to contents
        contents.push({
            role: message.role,
            parts: message.parts
        });
        break; // Done with user message


      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
  }

  // Finalize system_instruction
  let system_instruction = system_instruction_parts.length > 0 ? { parts: system_instruction_parts } : undefined;


  // Google API requires the first `contents` entry to be `user` if system_instruction is present,
  // or if there's no system instruction but history starts with model.
  // Ensure the very first actual content entry is 'user'.
  // This is a common requirement for turn-based models.
  if (contents.length > 0 && contents[0].role !== "user") {
      // Add a dummy user turn at the beginning if the history doesn't start with user.
      contents.unshift({ role: "user", parts: [{ text: "" }] });
  }

  // Clean up the `parts.callsMapping` helper property from parts arrays
  contents.forEach(content => {
      if (content.parts && Array.isArray(content.parts)) {
           delete content.parts.callsMapping;
      }
  });

  // console.info("Transformed Gemini Request Contents:", JSON.stringify(contents, null, 2));
  // console.info("Transformed Gemini Request System Instruction:", JSON.stringify(system_instruction, null, 2));

  return { system_instruction, contents };
};

// Transforms the 'content' field of an OpenAI message item (string or array)
// into an array of Gemini `parts`. Handles text, image_url, input_audio.
const transformMsgContent = async (message) => {
  const parts = [];
  const content = message.content; // Get the content field

  if (content === null || content === undefined || (Array.isArray(content) && content.length === 0)) {
      return parts; // Return empty parts array for null, undefined, or empty array content
  }

  if (!Array.isArray(content)) {
    // Simple case: content is a string (for user or assistant text messages)
    parts.push({ text: String(content) }); // Ensure it's a string part
    return parts;
  }

  // Complex case: content is an array of objects (for user messages with mixed content)
  let hasText = false;
  for (const item of content) {
    if (typeof item !== 'object' || item === null) {
        console.warn("Unexpected item type in message content array:", item);
        continue; // Skip invalid items
    }
    switch (item.type) {
      case "text":
        if (typeof item.text === 'string') {
            parts.push({ text: item.text });
            hasText = true; // Flag that text was found
        } else {
            console.warn("Invalid text part: text field is not a string.", item);
        }
        break;
      case "image_url":
        if (typeof item.image_url?.url === 'string') {
             try {
               const imgPart = await parseImg(item.image_url.url);
               parts.push(imgPart);
             } catch (err) {
                console.error("Failed to process image_url:", err);
                throw new HttpError("Failed to process image_url: " + err.message, 400);
             }
        } else {
             throw new HttpError("Invalid image_url part: missing 'url' string.", 400);
        }
        break;
      case "input_audio":
        if (typeof item.input_audio?.format === 'string' && typeof item.input_audio?.data === 'string') {
             parts.push({
                inlineData: {
                    mimeType: item.input_audio.format.startsWith("audio/") ? item.input_audio.format : `audio/${item.input_audio.format}`,
                    data: item.input_audio.data,
                }
             });
        } else {
             throw new HttpError("Invalid input_audio part: missing 'format' or 'data'.", 400);
        }
        break;
      // Add other known OpenAI content types if needed (e.g., image_file, etc.)
      default:
        console.warn(`Unknown or unsupported "content" item type (skipping): "${item.type}"`);
        // throw new HttpError(`Unknown "content" item type: "${item.type}"`, 400); // Or skip?
    }
  }

   // Google API sometimes requires a text part in a user turn if it contains other modalities (like images)
   // or if it contains *only* images/audio. Add an empty text part if no text was present initially.
  if (parts.length > 0 && !hasText) {
       // Double check if any part is actually a text part after processing
       const hasExistingTextPart = parts.some(p => p.text !== undefined);
       if (!hasExistingTextPart) {
           parts.push({ text: "" }); // Add an empty text part if none exists
       }
  }

  return parts;
};


const transformTools = (req) => {
  let tools, tool_config;
  if (req.tools && Array.isArray(req.tools)) {
    const funcs = req.tools.filter(tool => tool.type === "function");
    if (funcs.length > 0) {
         funcs.forEach(adjustSchema);
         tools = [{ function_declarations: funcs.map(tool => tool.function) }];
    } else {
         tools = []; // Send empty tools array if no function tools
    }
  } else if (req.tools !== undefined && req.tools !== null) {
       console.warn("Request 'tools' field is not an array. Ignoring.");
  }


  if (req.tool_choice !== undefined) {
    if (typeof req.tool_choice === "string") {
      const mode = req.tool_choice.toUpperCase();
      if (['NONE', 'AUTO', 'REQUIRED'].includes(mode)) {
         tool_config = { function_calling_config: { mode: mode } };
      } else {
         throw new HttpError(`Unsupported tool_choice string value: "${req.tool_choice}". Must be "none", "auto", or "required".`, 400);
      }
    } else if (typeof req.tool_choice === "object" && req.tool_choice !== null && req.tool_choice.type === "function") {
        const functionName = req.tool_choice.function?.name;
        if (typeof functionName === "string" && functionName) {
            tool_config = { function_calling_config: { mode: "ANY", allowed_function_names: [functionName] } };
        } else {
             throw new HttpError("Invalid tool_choice object: 'function.name' string is required for type 'function'.", 400);
        }
    } else {
       throw new HttpError("Invalid tool_choice format.", 400);
    }
  }

  return { tools, tool_config };
};


// Combines transformations for the full chat completions request body
const transformRequest = async (req) => {
    const { system_instruction, contents } = await transformMessages(req.messages);
    const generationConfig = transformConfig(req);
    const { tools, tool_config } = transformTools(req);

    const geminiRequestBody = {
        contents,
        ...(system_instruction && { system_instruction }),
        generationConfig,
        safetySettings, // Apply default safety settings
        ...(tools && tools.length > 0 && { tools }),
        ...(tool_config && { tool_config }),
    };

    // console.info("Final Gemini Chat Request Body:", JSON.stringify(geminiRequestBody, null, 2));

    return geminiRequestBody;
};


// --- Response Processing and Streaming ---

const generateId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = {
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
  "OTHER": "other",
   // Add others if needed, like "TOOL_FUNCTION_CALL" if Google API uses it as a distinct finishReason
};

// Transforms a single Gemini candidate object (from non-streaming response)
// into an OpenAI-like choice object for chat completions.
const transformMessageResponse = (cand) => {
  const message = {
      role: "assistant",
      content: [], // Content is an array of parts in OpenAI spec for multimodal
      // tool_calls: undefined
  };
  let hasFunctionCall = false;

  if (cand.content?.parts && Array.isArray(cand.content.parts)) {
      const tool_calls = [];
      const content_parts = []; // For text and image_url

      for (const part of cand.content.parts) {
          if (part.text !== undefined) {
              content_parts.push({ type: "text", text: part.text });
          } else if (part.inlineData !== undefined) {
               // Handle inline image data
               if (part.inlineData.mimeType && part.inlineData.data) {
                    // Gemini image response in chat comes as inlineData
                    content_parts.push({
                        type: "image_url",
                        image_url: {
                            url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`
                            // detail: "auto" // Optional
                        }
                    });
               } else {
                   console.warn("Skipping invalid inlineData part in candidate:", part.inlineData);
               }
          } else if (part.functionCall !== undefined) {
              // Handle function call part
              const fc = part.functionCall;
              if (fc.name && fc.args !== undefined) {
                   tool_calls.push({
                       id: fc.id ?? "call_" + generateId(), // Use Google ID or generate OpenAI-like
                       type: "function",
                       function: {
                           name: fc.name,
                           arguments: JSON.stringify(fc.args),
                       }
                   });
                   hasFunctionCall = true;
              } else {
                   console.warn("Skipping invalid functionCall part in candidate:", part.functionCall);
              }
          }
           // Ignore other part types (like functionResponse in a model response)
      }

      message.content = content_parts.length > 0 ? content_parts : null; // Use null if no text/image parts
      if (tool_calls.length > 0) {
          message.tool_calls = tool_calls; // Add tool_calls array if function calls were found
      }
  } else {
       // If no content or parts array, message content is null
       message.content = null;
  }


  let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || null;

  // If the message includes tool_calls, the finish reason is typically "tool_calls" in OpenAI format
  if (hasFunctionCall) {
      finish_reason = "tool_calls";
  }


  return {
    index: cand.index || 0, // Ensure index is present (default 0)
    message: message,
    logprobs: null, // Not available
    finish_reason: finish_reason,
    // original_finish_reason: cand.finishReason, // Optional
  };
};

// Transforms a single Gemini candidate object (from streaming chunk)
// into an OpenAI-like delta object for a stream chunk in chat completions.
const transformDeltaResponse = (cand) => {
    const delta = {
        // role: "assistant", // Role is typically only in the first delta chunk for a candidate
        // content: [], // Initialize delta content array (will be added if parts exist)
        // tool_calls: undefined // Initialize delta tool_calls array (will be added if parts exist)
    };
    let hasFunctionCall = false; // Flag if this chunk contains function calls

     if (cand.content?.parts && Array.isArray(cand.content.parts)) {
        const delta_tool_calls = [];
        const delta_content_parts = []; // For text and image_url

        for (const part of cand.content.parts) {
            if (part.text !== undefined) {
                 delta_content_parts.push({ type: "text", text: part.text });
            } else if (part.inlineData !== undefined) {
                 // Handle inline image data in delta (full part likely in one chunk)
                 if (part.inlineData.mimeType && part.inlineData.data) {
                     delta_content_parts.push({
                         type: "image_url",
                         image_url: {
                             url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`
                         }
                     });
                 } else {
                     console.warn("Skipping invalid inlineData part in stream chunk:", part.inlineData);
                 }
            } else if (part.functionCall !== undefined) {
                 // Handle function call part in delta (full call likely in one chunk)
                 const fc = part.functionCall;
                 if (fc.name && fc.args !== undefined) {
                     delta_tool_calls.push({
                          id: fc.id ?? "call_" + generateId(),
                          type: "function",
                          function: {
                              name: fc.name,
                              arguments: JSON.stringify(fc.args),
                          }
                     });
                     hasFunctionCall = true;
                 } else {
                      console.warn("Skipping invalid functionCall part in stream chunk:", part.functionCall);
                 }
            }
            // Ignore other part types
        }

        if (delta_content_parts.length > 0) {
            delta.content = delta_content_parts; // Add content array if parts exist
        }

        if (delta_tool_calls.length > 0) {
             delta.tool_calls = delta_tool_calls; // Add tool_calls array if function calls exist
        }
     }

    // Determine finish reason for this chunk (only present in the final chunk)
    let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || null;

    // Return the transformed choice delta object
    return {
        index: cand.index || 0, // Ensure index is present (default 0)
        delta: delta,
        finish_reason: finish_reason, // Includes finish reason if present in chunk
        // logprobs: null, // Not in delta
        // original_finish_reason: cand.finishReason, // Optional
    };
};


// Checks for prompt blocking feedback and adds a content filter choice if blocked
// This function is used for BOTH non-streaming (key="message") and streaming (key="delta")
const checkPromptBlock = (choices, promptFeedback, key) => {
  if (choices.length > 0) {
      return false; // Already have choices, not a full block scenario
  }

  if (promptFeedback?.blockReason) {
    console.log("Prompt was blocked by Google API. Reason:", promptFeedback.blockReason);
    if (promptFeedback.blockReason === "SAFETY" && promptFeedback.safetyRatings) {
      promptFeedback.safetyRatings
        .filter(r => r.blocked)
        .forEach(r => console.log(`- Safety Category: ${r.category}, Probability: ${r.probability}, Blocked: ${r.blocked}`));
    }

    // Add a single choice indicating content filtering
    choices.push({
      index: 0, // Always index 0 for the single blocked choice
      // Add the appropriate field based on whether it's streaming or non-streaming
      [key]: (key === "message") ? null : {}, // message is null, delta is empty object {}
      finish_reason: "content_filter", // Standard OpenAI reason
      // original_finish_reason: promptFeedback.blockReason, // Optional
    });
    return true; // Indicate that the prompt was blocked
  }
  return false; // Indicate that the prompt was not blocked
};


// Processes the full non-streaming response body from Google API for chat completions
// Transforms it into the OpenAI chat completion JSON string.
const processCompletionsResponse = (data, model, id) => {
  const obj = {
    id,
    choices: [], // Initialize choices array
    created: Math.floor(Date.now()/1000),
    // Report the model name without "models/" prefix, as typically done by OpenAI
    model: model.startsWith("models/") ? model.substring(7) : model,
    // system_fingerprint: data.system_fingerprint ?? "fp_unknown",
    object: "chat.completion", // Standard OpenAI object type
    // usage: ... // Added after checking for prompt block
  };

  // Process candidates if they exist
  if (data.candidates && Array.isArray(data.candidates)) {
     obj.choices = data.candidates.map(transformMessageResponse);
  } else {
      console.warn("Google API response missing 'candidates' array for non-streaming chat completion.");
      obj.choices = []; // Ensure choices is an array even if missing
  }


  // Check for prompt blocking *after* attempting to transform candidates.
  // If the prompt was blocked, this function will add a content_filter choice.
  checkPromptBlock(obj.choices, data.promptFeedback, "message");


  // Add usage metadata if available
  if (data.usageMetadata) {
    obj.usage = transformUsage(data.usageMetadata);
  } else if (data.promptFeedback?.tokenCount) {
      // If prompt was blocked, prompt token count might be in promptFeedback
       obj.usage = { prompt_tokens: data.promptFeedback.tokenCount, completion_tokens: 0, total_tokens: data.promptFeedback.tokenCount };
  } else {
      obj.usage = undefined; // or null;
  }

  return JSON.stringify(obj); // Compact JSON output
};

// Transforms Google API usageMetadata to OpenAI usage object
const transformUsage = (data) => {
  const usage = {
    prompt_tokens: data.promptTokenCount ?? 0,
    completion_tokens: data.candidatesTokenCount ?? 0,
    total_tokens: data.totalTokenCount ?? 0,
  };
   // Ensure fields are numbers
   Object.keys(usage).forEach(key => { usage[key] = Number(usage[key]) || 0; });
   return usage;
};


// --- Streaming Transform Functions ---

// Regex to parse Server-Sent Events 'data:' lines
const responseLineRE_single = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;

// TransformStream for parsing SSE data chunks.
function parseStream (chunk, controller) {
  this.buffer += chunk;
  let match;
  while ((match = this.buffer.match(responseLineRE_single)) !== null) {
    controller.enqueue(match[1]);
    // Important: Use slice on the underlying buffer and convert back to string for efficiency
    this.buffer = this.buffer.buffer.slice(match[0].length).toString();
  }
}

// Flush function for parseStream. Handles any remaining data.
function parseStreamFlush (controller) {
  if (this.buffer.length > 0) {
    console.error("parseStream: Remaining data in buffer after flush:", this.buffer);
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true; // Indicate leftover data
  }
  this.buffer = ""; // Clear buffer
}

// Delimiter for SSE messages
const delimiter = "\n\n";

// Helper to format an object into an SSE 'data:' line
const sseline = (obj) => {
  const dataString = JSON.stringify(obj);
  return `data: ${dataString}${delimiter}`;
};

// TransformStream for transforming Google API stream chunks (parsed JSON)
// into OpenAI API stream chunks (SSE 'data:' lines) for chat completions.
function toOpenAiStream (line, controller) {
  let data;
  try {
    data = JSON.parse(line);
    // Check for Google API errors embedded in the stream
    if (data.error) {
        console.error("Google API stream error chunk:", data.error);
        const errorChunk = {
            id: this.id,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now()/1000),
            model: this.model.startsWith("models/") ? this.model.substring(7) : this.model,
            choices: [{
                index: 0,
                delta: { role: "assistant" },
                finish_reason: "tool_error", // Or map a more specific error reason?
            }],
            error: { // Include error details in the chunk
                message: data.error.message || "An error occurred during streaming.",
                type: data.error.status || "api_error",
                code: data.error.code,
             },
        };
        controller.enqueue(sseline(errorChunk));
        controller.enqueue("data: [DONE]" + delimiter); // Signal end of stream
        return;
    }
     // A valid chunk fragment should typically have 'candidates' or 'promptFeedback'
    if (!data.candidates && !data.promptFeedback) {
       if (this.shared.is_buffers_rest && line.trim().length > 0) {
           console.warn("toOpenAiStream: Ignoring potentially incomplete or malformed chunk:", line);
       } else {
           console.warn("toOpenAiStream: Unexpected Google API stream chunk format:", data);
           // Decide whether to throw or skip. Skipping is more resilient.
       }
       return; // Skip processing this chunk
    }

  } catch (err) {
    console.error("toOpenAiStream: Error parsing JSON line:", err, "Line:", line);
    if (!this.shared.is_buffers_rest || line.trim().length > 0) {
        // Throwing is usually better for signaling critical stream errors
        controller.error(new Error("Failed to parse stream chunk JSON: " + err.message));
        // Or enqueue an error chunk and DONE... decide on error handling strategy
    }
    return; // Skip if it was just leftover buffer
  }

  // Handle prompt blocking feedback received in a chunk
  if (data.promptFeedback) {
      const choices = [];
      if (checkPromptBlock(choices, data.promptFeedback, "delta")) {
          const blockChunk = {
             id: this.id,
             object: "chat.completion.chunk",
             created: Math.floor(Date.now()/1000),
             model: this.model.startsWith("models/") ? this.model.substring(7) : this.model,
             choices: choices, // Contains the single blocked choice ({ index: 0, delta: {}, finish_reason: "content_filter" })
             // Usage might be included here
             usage: data.promptFeedback.tokenCount ? { prompt_tokens: data.promptFeedback.tokenCount, completion_tokens: 0, total_tokens: data.promptFeedback.tokenCount } : undefined,
          };
           controller.enqueue(sseline(blockChunk));
           controller.enqueue("data: [DONE]" + delimiter); // Signal end of stream
           return;
      }
       // If promptFeedback is present but not a block, and no candidates, skip
       if (!data.candidates) {
           console.warn("toOpenAiStream: Chunk contains promptFeedback but no candidates and is not a block.", data);
           return;
       }
  }


  // Process candidates (should be an array, typically with 1 candidate in streaming)
  if (data.candidates && Array.isArray(data.candidates)) {
    const openaiChoices = data.candidates.map(transformDeltaResponse);

    openaiChoices.forEach(choice => {
        const index = choice.index || 0;
        // Initialize state for this candidate index if it's the first chunk for it
        if (!this.last[index]) {
            this.last[index] = { delta: {}, finish_reason: null };
        }

        // OpenAI stream format: role only in the very first chunk (index 0, first in stream)
        if (index === 0 && Object.keys(this.last[index].delta).length === 0 && !choice.delta.role) {
             choice.delta = choice.delta || {}; // Ensure delta exists
             choice.delta.role = "assistant"; // Add role to the first chunk of the first candidate
        }

        // Update accumulated state for finish reason
        if (choice.finish_reason) {
            this.last[index].finish_reason = choice.finish_reason;
        }

        // Enqueue the transformed OpenAI chunk
        const openaiChunk = {
           id: this.id,
           object: "chat.completion.chunk",
           created: Math.floor(Date.now()/1000),
           model: this.model.startsWith("models/") ? this.model.substring(7) : this.model,
           choices: [choice], // Each OpenAI chunk has a 'choices' array with one item
           // Usage metadata is typically in the final chunk. Add it here if requested and available.
           usage: data.usageMetadata && this.streamIncludeUsage ? transformUsage(data.usageMetadata) : undefined,
        };

         // Only enqueue if the delta has content/tool_calls, or it's the first chunk (role), or it has finish reason/usage
         const deltaHasMeaningfulContent = choice.delta && (Object.keys(choice.delta).length > 0 || choice.delta.role);
         const chunkHasFinishReasonOrUsage = choice.finish_reason || openaiChunk.usage;

         if (deltaHasMeaningfulContent || chunkHasFinishReasonOrUsage) {
             controller.enqueue(sseline(openaiChunk));
         } else {
             // console.log(`Skipping empty chunk for index ${index}:`, openaiChunk);
         }

    });

  } else {
      console.warn("toOpenAiStream: Chunk missing 'candidates' array and not a prompt block:", data);
      // Skip processing this chunk
  }
}

// Flush function for toOpenAiStream. Sends the final [DONE] signal.
function toOpenAiStreamFlush (controller) {
  // After processing all chunks, send the final [DONE] signal.
  controller.enqueue("data: [DONE]" + delimiter);

  // Clean up state (optional, but good practice)
  this.last = [];
  this.shared = {};
}

// --- END OF RESPONSE PROCESSING AND STREAMING ---
