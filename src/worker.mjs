--- START OF FILE worker.mjs ---

import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
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
  // Add other common headers if needed, e.g., Access-Control-Allow-Headers, Access-Control-Allow-Methods
  // headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
  // headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*", // Allow all methods for simplicity, adjust as needed
      "Access-Control-Allow-Headers": "*", // Allow all headers for simplicity, adjust as needed
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
  // Clone the response for potential later use (e.g., reading text body)
  const responseClone = response.clone();
  let responseBodyText;

  try {
    responseBodyText = await responseClone.text();
  } catch (e) {
    // Handle cases where response body might not be readable as text
    console.error("Error reading response text:", e);
    // Fallback, try to return the original response body if possible
    return new Response(response.body, fixCors(response));
  }

  if (response.ok) {
    try {
      const { models } = JSON.parse(responseBodyText);
      // Filter out vision models that might not be intended for chat/completions/embeddings proxy
      // Keep only models that are likely relevant, or transform all.
      // For simplicity, transforming all listed models for now.
      body = JSON.stringify({
        object: "list",
        data: models.map(({ name }) => ({
          id: name.replace("models/", ""), // Remove 'models/' prefix
          object: "model",
          created: 0, // Static, as original API doesn't provide timestamp
          owned_by: "google", // Or original owner if available/desired
        })),
        // Add pagination or other fields if needed, based on OpenAI spec
      }, null, "  "); // Pretty print for debugging
    } catch (err) {
      console.error("Error processing models response:", err);
      // If transformation fails, return the original (potentially error) response body
      return new Response(responseBodyText, fixCors(response));
    }
  } else {
    // If the original response was not ok, return its body (likely an API error message)
    body = responseBodyText;
  }

  // Return the transformed body or the original error body with CORS headers
  return new Response(body, fixCors(response));
}


const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004"; // Use a modern default
async function handleEmbeddings (req, apiKey) {
  if (typeof req.model !== "string") {
    // Suggest a default model if none is specified, or throw error?
    // OpenAI spec typically requires model. Let's require it or use a strong default.
    // For now, sticking to original logic which assumes a model might be missing but handles it.
    // Throwing is safer for API compliance.
    // throw new HttpError("model is not specified", 400);
    // Reverting to original code's logic: use default if not string.
  }

  let model;
  // Use model provided in request, prioritizing 'models/' prefix if present
  if (req.model?.startsWith("models/")) {
    model = req.model;
  } else {
    // If no 'models/' prefix, assume just the model name.
    // If it doesn't look like a Gemini model (like gemini-, gemma-, learnlm-), default it.
    // The original code defaulted if it *wasn't* gemini-*. Let's correct that logic.
    // It should default *unless* it looks like a valid model name fragment Google uses.
    const validModelRegex = /^(gemini|gemma|text-embedding|learnlm)-.*/;
    if (!req.model || !validModelRegex.test(req.model)) {
         req.model = DEFAULT_EMBEDDINGS_MODEL; // Use the default embedding model
    }
    model = "models/" + req.model; // Prepend 'models/'
  }


  // Input can be a string or array of strings
  if (!Array.isArray(req.input)) {
    // Original code wrapped a single string in an array. Keep this behavior.
    req.input = [ req.input ];
  }

  // API endpoint is batchEmbedContents for multiple inputs
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        // model needs to be included in each request for batch endpoint
        model,
        content: { parts: [{ text }] }, // Input is text parts
        outputDimensionality: req.dimensions, // Optional: specify dimension
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
      const { embeddings } = JSON.parse(responseBodyText);
      if (!Array.isArray(embeddings)) {
          throw new Error("Unexpected response format: embeddings array not found.");
      }
      body = JSON.stringify({
        object: "list",
        data: embeddings.map(({ values }, index) => ({
          object: "embedding",
          index, // Original index from the input request order
          embedding: values,
        })),
        model: req.model, // Report the requested model name (without 'models/' prefix)
        // usage: ... // Add usage if available from the API response
      }, null, "  ");
    } catch (err) {
        console.error("Error processing embeddings response:", err);
        return new Response(responseBodyText, fixCors(response));
    }
  } else {
     body = responseBodyText;
  }
  return new Response(body, fixCors(response));
}


const DEFAULT_MODEL = "gemini-2.0-flash"; // Default chat model
async function handleCompletions (req, apiKey) {
  let model = DEFAULT_MODEL;
  // Determine the model name without the 'models/' prefix for the API call URL
  // and for reporting in the response object.
  if (typeof req.model === "string") {
      if (req.model.startsWith("models/")) {
          model = req.model.substring(7); // Remove 'models/' prefix
      } else {
          // Assume model name is directly provided (e.g., "gemini-pro", "gemini-2.0-flash-exp-image-generation")
          model = req.model;
      }
  }
  // Note: The actual API call URL will be BASE_URL/API_VERSION/models/{model_name}:TASK
  // where model_name is just "gemini-pro", "gemini-2.0-flash", etc.

  let body = await transformRequest(req); // Transforms OpenAI -> Gemini request body

  // Handle specific model suffix requests like "-search-preview" or ":search"
  // This modifies the request body to include Google Search tool.
  // Note: This is specific to certain Gemini models/features.
  const originalModelReq = req.model?.toLowerCase() || "";
  const isSearchModel = originalModelReq.endsWith(":search") || originalModelReq.endsWith("-search-preview");

  if (isSearchModel) {
      // Remove the search suffix from the model name sent to the API if present
      // The Google API endpoint is models/gemini-pro:generateContent, not models/gemini-pro:search:generateContent
      if (model.endsWith(":search")) {
          model = model.substring(0, model.length - 7);
      } else if (model.endsWith("-search-preview")) {
          model = model.substring(0, model.length - 16);
      }
      body.tools = body.tools || [];
      // Ensure the tool is only added once if it's already there or if multiple search models are specified
      if (!body.tools.some(tool => tool.googleSearch)) {
         body.tools.push({googleSearch: {}});
      }
  }


  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) {
      url += "?alt=sse"; // Append alt=sse for Server-Sent Events streaming format
      // For stream_options?.include_usage, Google API adds usage metadata within the stream chunks.
      // We handle this during the stream transformation.
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
          last: [], // State for accumulating/tracking last chunk info per candidate (though less crucial with array content)
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
           // Re-throw as HttpError or return a specific response based on data.error
           // For now, return the original error body.
           responseBody = responseBodyText; // Keep the original error JSON
           // Also fix the status code if data.error.code is available and standard HTTP
           if (data.error.code && typeof data.error.code === 'number') {
               response = new Response(responseBody, fixCors({ status: data.error.code, headers: response.headers }));
               // Need to return here to avoid processing as a successful response
               return response;
           }

        } else if (!data.candidates) {
          // Successful response should have candidates array (even if empty)
          // If not, it's an unexpected format
          throw new Error("Invalid completion object: 'candidates' array not found.");
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
  }

  // Return the final Response object with the appropriate body and CORS headers.
  // fixCors ensures CORS headers are present regardless of success/failure status.
  return new Response(responseBody, fixCors(response));
}


// Helper to adjust schema properties for compatibility (e.g., remove additionalProperties: false)
const adjustProps = (schemaPart) => {
  if (typeof schemaPart !== "object" || schemaPart === null) {
    return;
  }
  if (Array.isArray(schemaPart)) {
    schemaPart.forEach(adjustProps);
  } else {
    // If it's an object with properties and specifically set additionalProperties to false, remove it.
    // Some schemas might use this in ways Gemini doesn't strictly support or expects differently.
    if (schemaPart.type === "object" && schemaPart.properties && schemaPart.additionalProperties === false) {
      delete schemaPart.additionalProperties;
    }
    // Recursively process all values (properties in an object, items in an array)
    Object.values(schemaPart).forEach(adjustProps);
  }
};
// Helper to adjust the overall function schema structure
const adjustSchema = (tool) => {
  // Assuming tool is like { type: "function", function: { name, description, parameters: { type: "object", properties, required } } }
  if (!tool || tool.type !== "function" || !tool.function || !tool.function.parameters) {
      return; // Not a function tool with parameters
  }
  const parameters = tool.function.parameters;
  // Remove 'strict' or other potentially unsupported top-level schema properties
  // Google's API expects the schema structure directly under 'parameters'.
  // Let's focus on adjusting the content *within* the schema if needed.
  // Example: original code `const obj = schema[schema.type]; delete obj.strict; return adjustProps(schema);`
  // This assumes schema is like { type: "object", ... }. OpenAI schema is under parameters.
  // Let's adjust the parameters object directly.
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
  // Add or remove categories as supported/needed by the target Gemini model
  // "HARM_CATEGORY_CIVIC_INTEGRITY", // Some models may not support all
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE", // Defaulting to BLOCK_NONE for maximum output, adjust if stricter filtering is desired
}));

// Mapping OpenAI request fields to Gemini generationConfig fields
const fieldsMap = {
  // Standard OpenAI fields mapped to Gemini
  frequency_penalty: "frequencyPenalty",
  max_tokens: "maxOutputTokens", // OpenAI max_tokens maps to Gemini maxOutputTokens
  presence_penalty: "presencePenalty",
  // OpenAI fields not directly mapped or handled differently:
  // prompt: Handled by messages transformation
  // logit_bias: Not directly supported in Gemini generationConfig
  // user: Not directly supported in Gemini generationConfig (handled in messages if needed contextually)
  // stop: Handled directly as stopSequences in generationConfig
  // stream: Handled by choosing endpoint (streamGenerateContent)
  // model: Handled before calling transformRequest
  // temperature, top_p: Already match name "temperature", "topP"

  // Custom/Non-standard fields often used, mapped to Gemini
  seed: "seed", // Used for reproducible generation if supported
  top_k: "topK", // Gemini supports topK
  // n: OpenAI 'n' (number of candidates) maps to Gemini 'candidateCount'.
  // Gemini API `generateContent` supports `candidateCount` in `generationConfig`.
  // However, `streamGenerateContent` *only* supports `candidateCount=1`.
  // Need to handle `n` based on streaming or non-streaming.
  // Let's handle 'n' directly in handleCompletions or generationConfig transformation.
  // The existing transformConfig does this. Let's keep it.
  candidateCount: "candidateCount", // Added explicitly based on original code's use
};

const transformConfig = (req) => {
  let cfg = {};
  // Directly map fields using the map
  for (const openAiKey in fieldsMap) {
    const geminiKey = fieldsMap[openAiKey];
    if (req[openAiKey] !== undefined) { // Only map if the key exists in the request
      cfg[geminiKey] = req[openAiKey];
    }
  }

  // Handle stop sequences: OpenAI 'stop' can be string or array. Gemini 'stopSequences' is array.
  if (req.stop !== undefined) {
      cfg.stopSequences = Array.isArray(req.stop) ? req.stop : [req.stop];
  }
  // Handle n (candidateCount): Only apply if not streaming, as streaming only supports 1.
  if (req.n !== undefined && !req.stream) {
      cfg.candidateCount = req.n;
  }


  // Handle response_format for JSON mode, etc.
  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        // OpenAI's json_schema includes the full schema object.
        // Gemini expects responseSchema as the schema object and responseMimeType.
        // The schema object itself might need adjustments (e.g., removing additionalProperties: false).
        if (!req.response_format.json_schema?.schema) {
             throw new HttpError("json_schema response_format requires a 'schema' object.", 400);
        }
        // Recursively adjust properties within the provided schema
        adjustProps(req.response_format.json_schema.schema);
        cfg.responseSchema = req.response_format.json_schema.schema;
        // If the top level schema is just an enum, use text/x.enum mime type as per some examples
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break; // Specific handling for enum schema
        }
        // Otherwise, fall through to treat as generic application/json schema
        // eslint-disable-next-line no-fallthrough
      case "json_object":
        // OpenAI's json_object simply requires JSON output.
        // Gemini uses responseMimeType: "application/json".
        cfg.responseMimeType = "application/json";
        // Ensure a schema exists for json_object if the model requires it, although for basic json_object it might not be strictly needed by Gemini.
        // If transforming 'json_object' to 'json_schema' with an empty schema is necessary, add logic here.
        // For now, just setting mime type.
        break;
      case "text":
        // OpenAI's text response format.
        // Gemini uses responseMimeType: "text/plain".
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
      // Determine MIME type from Content-Type header
      mimeType = response.headers.get("content-type");
      if (!mimeType || !mimeType.startsWith('image/')) {
          // If no mime type or not an image type, try to infer or error
           console.warn(`Warning: Fetched URL (${url}) did not return an image Content-Type (${mimeType || 'none'}). Attempting to process.`);
           // Fallback: try common image types or require it
           // For simplicity, proceed hoping the buffer/data URL conversion works or require a known type
           if (!mimeType) mimeType = 'application/octet-stream'; // Default if unknown
      }
      // Read as ArrayBuffer and convert to Base64 string
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      // Wrap fetch errors
      throw new Error("Error fetching image from URL: " + err.message);
    }
  } else if (url.startsWith("data:")) {
    // Parse Data URL
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match || !match.groups?.mimeType || !match.groups?.data) {
      throw new HttpError("Invalid image data URL format.", 400);
    }
    ({ mimeType, data } = match.groups);
    // Ensure mimeType is an image type if we want to be strict
     if (!mimeType.startsWith('image/')) {
         console.warn(`Warning: Data URL does not specify an image MIME type (${mimeType}). Attempting to process.`);
     }
     // Check if base64 is specified and data looks base64 encoded
     if (match.groups[2] !== ';base64' && !/^[a-zA-Z0-9+/=]+$/.test(data)) {
         console.warn("Warning: Data URL appears not to be base64 encoded but is missing ';base64'. Proceeding anyway.");
         // Depending on requirements, could try to decode or throw an error
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

// Transform OpenAI 'tool' role message to Gemini 'function' role content part
const transformFnResponse = ({ content, tool_call_id }, previousMessageParts) => {
  // The 'tool' role message in OpenAI corresponds to the model's response to a function call.
  // This response needs to be added as a 'functionResponse' part in the *next* user turn message.
  // The OpenAI 'tool' message contains the function call *ID* it is responding to,
  // and the 'content' which is the result of the function execution (stringified JSON).

  if (!tool_call_id) {
    throw new HttpError("tool_call_id not specified for tool message.", 400);
  }

  let responseData;
  try {
    // The content of a tool message is the stringified JSON result from the tool.
    responseData = JSON.parse(content);
  } catch (err) {
    console.error("Error parsing function response content:", err);
    throw new HttpError("Invalid function response content (not valid JSON): " + content, 400);
  }

  // The Google API needs the *name* of the function that was called, not just the ID.
  // We need to find the original function call in the *previous* assistant message.
  // The original code stored this mapping in `parts.calls` when transforming the assistant message.
  // This state management between turns is tricky. A simpler approach might be to require
  // the function name in the tool message content itself, or rely on the client to provide it.
  // The original code assumes `previousMessageParts` (the parts of the user message we are building)
  // contains `calls` information from the *preceding* assistant message. This seems incorrect.
  // The `calls` mapping should likely be stored/passed down per conversation turn, or the
  // OpenAI tool message should include the function name (it doesn't).

  // Let's modify this based on the likely intent: the tool message needs to know which
  // function call it's responding to from the *immediately preceding* assistant message.
  // This worker doesn't maintain conversation state. It only processes the current request.
  // The `messages` array in the request *does* contain the history.
  // We need to find the corresponding `tool_calls` in the previous `assistant` message.

  // To correctly handle this without global state, the transformation needs access to the *entire* messages array history
  // or the tool message format needs to include the function name. OpenAI spec doesn't put name in tool message.
  // The most robust way is to look back in the message history.

  // Re-thinking: The original code's `transformMessages` iterates through messages.
  // When it hits a `tool` message, it attempts to add its response to the *current* user's `parts` array.
  // It assumes the previous assistant message's tool calls info is available via `parts.calls`. This is flawed.
  // Let's fix the `transformMessages` structure to handle this correctly.

  // (This function will be refactored into `transformMessages`)
  // return {
  //   functionResponse: {
  //     // id: tool_call_id.startsWith("call_") ? null : tool_call_id, // Google API doesn't seem to use/need the call ID here? Docs show just `name` and `response`.
  //     name: functionName, // Need the function name here!
  //     response: responseData,
  //   }
  // };
};

// Transform OpenAI assistant 'tool_calls' into Gemini 'functionCall' parts.
// This is used when transforming the assistant's response *into* Gemini format
// (this is incorrect, this should be used when transforming the *request* messages,
// specifically the assistant's previous turn *if* it included tool calls that the user is now responding to).
// The original code seems to apply this transformation to the assistant message *in the request history*.
// Let's adjust its placement to be used within `transformMessages`.
const transformFnCalls = ({ tool_calls }) => {
  // When transforming an assistant message that includes tool_calls in the OpenAI format request
  // to the Gemini format request, these tool_calls become `functionCall` parts in the model's content.
  // We also need to store the mapping from tool_call.id to function.name for subsequent 'tool' messages.
  const callsMapping = {}; // Map tool_call_id -> function.name
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type }, i) => {
    if (type !== "function") {
      // Gemini primarily supports function calls via tools.
      throw new HttpError(`Unsupported tool_call type in assistant message history: "${type}". Only "function" is supported.`, 400);
    }
    let args;
    try {
      // Arguments are stringified JSON in OpenAI format
      args = JSON.parse(argstr);
    } catch (err) {
      console.error("Error parsing function arguments in assistant message:", err);
      throw new HttpError("Invalid function arguments in assistant message history: " + argstr, 400);
    }
    // Store the mapping for potential subsequent 'tool' role messages
    callsMapping[id] = name; // Map OpenAI tool_call_id to Gemini function name
    return {
      functionCall: {
        // Gemini API uses 'name' and 'args'. The 'id' from OpenAI tool_call isn't directly mapped here.
        // If Google API supports an ID for the call itself, it's not commonly documented/used in functionCall part.
        // Let's omit the ID unless Google API specifically supports it here.
        // Original code included id: id.startsWith("call_") ? null : id, which might be an attempt to map it.
        // Let's keep the ID mapping in `callsMapping` but not add it to the `functionCall` part itself.
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
      // Return empty content or handle error if messages are required
      if (messages === undefined) return { contents: [] };
      throw new HttpError("messages must be an array.", 400);
  }

  const contents = [];
  let system_instruction;
  let lastAssistantToolCallsMapping = {}; // Store mapping from the previous assistant message

  for (const item of messages) {
    // Deep clone the message item to avoid modifying the original request object
    const message = JSON.parse(JSON.stringify(item));

    switch (message.role) {
      case "system":
        // Gemini's system instruction is separate, not part of the contents array.
        // It should be the first message in the OpenAI list. If multiple, concatenate?
        // Google API v1beta expects a single `system_instruction: { parts: [...] }`.
        // If multiple system messages are provided, concatenate their content.
        const systemParts = await transformMsgContent(message);
         if (system_instruction) {
             // Concatenate system instructions if multiple are provided
             system_instruction.parts.push(...systemParts);
         } else {
             system_instruction = { parts: systemParts };
         }
        continue; // Skip adding system message to contents array

      case "tool": {
        // An OpenAI 'tool' message is the result of a function call executed by the user/client.
        // This corresponds to a 'functionResponse' part in the *user's* content turn.
        // It must immediately follow an assistant message with `tool_calls`.
        // It requires the `tool_call_id` it is responding to, and the `content` result.

        if (!message.tool_call_id) {
             throw new HttpError("tool_call_id is required for messages with role 'tool'.", 400);
        }
        // We need the function name associated with this tool_call_id.
        // This mapping was provided in the *immediately preceding* assistant message's tool_calls.
        const functionName = lastAssistantToolCallsMapping[message.tool_call_id];
        if (!functionName) {
             throw new HttpError(`Function name not found for tool_call_id "${message.tool_call_id}". Ensure the preceding assistant message contained this tool_call_id.`, 400);
        }

        let responseData;
        try {
             // Content of a tool message is the stringified JSON result.
            responseData = JSON.parse(message.content);
        } catch (err) {
             console.error("Error parsing tool message content:", err);
             throw new HttpError("Invalid content in tool message (not valid JSON).", 400);
        }

        // Add the functionResponse part to the *last* content entry, which should be the user's turn.
        const lastContent = contents[contents.length - 1];
        if (!lastContent || lastContent.role !== "user") {
             // This shouldn't happen in a valid conversation turn (user message with tool response follows assistant tool call)
            throw new HttpError("Tool message must be immediately preceded by a user message.", 400);
             // Or, more accurately, the tool message *itself* is the user's response, and its parts
             // should be added to the user's content entry.
             // Let's assume the 'tool' message IS the user's turn responding to the assistant's call.
             // The structure should be Assistant(tool_calls) -> User(tool_response).
             // So, we add the functionResponse part to the *current* user content we are building.
             // But the code iterates message by message. A tool message *is* a message entry.
             // It seems the Google API wants the function response in the *same* content block as the user's message parts.
             // Example: User: {parts: [{text: "What's the weather like?"}]}, Model: {parts: [{functionCall: ...}]}, User: {parts: [{functionResponse: ...}, {text: "And what about tomorrow?"}]}

             // Correct approach based on Google examples: A user turn can have text parts and functionResponse parts.
             // An assistant turn can have text parts, image parts, and functionCall parts.
             // The `messages` array in the request represents these turns.
             // So, a `tool` message maps to a 'user' role content entry with a `functionResponse` part.
             // It seems a tool message in OpenAI *only* has `role: tool`, `tool_call_id`, and `content`.
             // It doesn't have additional text parts from the user for that turn.
             // So, a tool message `item` needs to become a `contents` entry with `role: "user"`
             // and a single `functionResponse` part.

            contents.push({
                role: "user", // Tool response is considered part of the user's turn
                parts: [{
                     functionResponse: {
                        name: functionName, // Function name found from the previous assistant message's tool_calls mapping
                        response: responseData, // The JSON response from the tool
                        // The ID from OpenAI tool_call_id could potentially be included here if Google API supports it, but docs don't show it.
                     }
                }]
                // Note: OpenAI tool messages don't have 'content' array or 'tool_calls', just 'content' string.
                // So, no other parts like text or images should come from this specific message item.
            });

        }
        continue; // Skip default processing below
      }

      case "assistant":
        // Transform role for Google API
        message.role = "model";
         // Process content and tool_calls for this assistant message
        if (message.tool_calls && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
             // Transform OpenAI tool_calls into Gemini functionCall parts
             const toolCallParts = transformFnCalls(message);
             message.parts = toolCallParts;
             // Store the mapping from this assistant message's tool calls for the *next* tool message
             lastAssistantToolCallsMapping = toolCallParts.callsMapping || {};
             // Assistant message with tool_calls should not have 'content' string per OpenAI spec
             // If it does, merge or handle error? Let's prioritize tool_calls if present.
             if (message.content !== undefined && message.content !== null && message.content !== "") {
                 console.warn("Assistant message has both tool_calls and content. Prioritizing tool_calls.");
                 // Optional: transform content string to text part and add it?
                 // message.parts.unshift({ text: message.content });
             }
             delete message.content; // Remove the OpenAI 'content' string field

        } else {
             // Process standard text/image content for assistant message
             message.parts = await transformMsgContent(message);
             // Assistant message without tool_calls must have content
             if (!message.parts || message.parts.length === 0) {
                 // OpenAI spec says 'content' is required unless tool_calls are specified.
                 // If no tool_calls and no content, it's an empty assistant message, which might be okay or an error.
                 // Google API might expect at least one part.
                 console.warn("Empty assistant message content received.");
                 // Add a default empty text part? Or let Google API handle validation? Let's let API validate.
             }
             lastAssistantToolCallsMapping = {}; // Reset mapping as this message has no tool calls
             delete message.content; // Remove the OpenAI 'content' string field
        }
        break; // Add this transformed message to contents

      case "user":
        // Role is already correct for Google API ('user')
         // Process content for user message
        message.parts = await transformMsgContent(message);
         if (!message.parts || message.parts.length === 0) {
             // User message must have content parts (text, image, function_response etc.)
             // Add a minimal part if empty to prevent API errors? Or throw?
             // Google API requires user turns to have at least one part.
             // Let's add an empty text part if content is completely empty.
             if (!message.content || (Array.isArray(message.content) && message.content.length === 0)) {
                 message.parts = [{ text: "" }]; // Minimal text part
                 console.warn("Empty user message content received, adding empty text part.");
             }
         }
         // If the *previous* assistant message had tool calls, and this user message
         // includes a function response part (transformed from a 'tool' role message
         // which was already added in the 'tool' case above), that function response part
         // is already handled. We just need to ensure any text/image parts from *this*
         // user message (if it had a complex content array in OpenAI format) are also added.
         // The `transformMsgContent` handles the text/image parts from the user's 'content' field.
         // Function responses from 'tool' role messages are handled in the 'tool' case.
         // If a user message *itself* contains function_response parts directly in its 'content' array
         // in OpenAI format, the current `transformMsgContent` won't handle it.
         // Let's assume OpenAI 'tool' role is the standard way to provide function responses.
         // So, for a 'user' role message item, we only process its 'content' for text/images.
         delete message.content; // Remove the OpenAI 'content' field string/array
        lastAssistantToolCallsMapping = {}; // Reset mapping as this is a user message
        break; // Add this transformed message to contents

      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }

    // Add the transformed message (with 'parts' array) to the contents array
    // Only add if it's a user or model turn (not system)
    if (message.role === "user" || message.role === "model") {
       contents.push({
           role: message.role,
           parts: message.parts // Add the transformed parts array
       });
    }
  }

  // Google API requires that `system_instruction` is the very first item
  // *before* any `contents`. It also requires the first `contents` entry to be `user`.
  // If the first actual content entry is `model` (e.g., history starts with assistant),
  // Google API might require a placeholder user turn.
  // The original code added an empty user turn if system_instruction existed and first content was not text.
  // Let's ensure the first content role is 'user'.
  if (contents.length > 0 && contents[0].role !== "user") {
      // Add a dummy user turn if the history doesn't start with user.
      // This is a common requirement for turn-based models.
      contents.unshift({ role: "user", parts: [{ text: "" }] });
  }

  // Remove the `parts.callsMapping` helper property from the parts arrays before sending to Google
  contents.forEach(content => {
      if (content.parts && Array.isArray(content.parts)) {
           delete content.parts.callsMapping; // Clean up the helper property
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
      // Handle empty content, return empty parts array.
      return parts;
  }

  if (!Array.isArray(content)) {
    // Simple case: content is a string (for user or assistant text messages)
    parts.push({ text: String(content) }); // Ensure it's a string
    return parts;
  }

  // Complex case: content is an array of objects (for user messages with mixed content)
  // User messages can have array content with text, image_url, input_audio in OpenAI format.
  let hasText = false;
  for (const item of content) {
    if (typeof item !== 'object' || item === null) {
        // Handle unexpected item types in the array, perhaps convert to text or skip?
        console.warn("Unexpected item type in message content array:", item);
        continue; // Skip invalid items
    }
    switch (item.type) {
      case "text":
        if (typeof item.text === 'string') {
            parts.push({ text: item.text });
            hasText = true;
        } else {
            console.warn("Invalid text part: text field is not a string.", item);
        }
        break;
      case "image_url":
        if (typeof item.image_url?.url === 'string') {
             try {
               // parseImg fetches or processes data URL and returns { inlineData: { ... } }
               const imgPart = await parseImg(item.image_url.url);
               parts.push(imgPart);
               // Note: Google API image parts don't count as text parts for the "must have text" rule.
             } catch (err) {
                console.error("Failed to parse image_url:", err);
                // Depending on strictness, either throw or skip the image and warn.
                // Throwing is safer for API proxying errors early.
                throw new HttpError("Failed to process image_url: " + err.message, 400);
             }
        } else {
             throw new HttpError("Invalid image_url part: missing 'url' string.", 400);
        }
        break;
      case "input_audio":
        // Check for required fields and format
        if (typeof item.input_audio?.format === 'string' && typeof item.input_audio?.data === 'string') {
             parts.push({
                inlineData: {
                    // Ensure mimeType is correctly formatted, e.g., "audio/wav", "audio/mpeg"
                    mimeType: item.input_audio.format.startsWith("audio/") ? item.input_audio.format : `audio/${item.input_audio.format}`,
                    data: item.input_audio.data, // Base64 data
                }
             });
        } else {
             throw new HttpError("Invalid input_audio part: missing 'format' or 'data'.", 400);
        }
        break;
      case "tool_code":
        // This is an input part type in Gemini. Map it if supported/needed.
        // OpenAI 'tool' role is for *response* to a call. This seems different.
        // Assuming not standard for OpenAI chat completions proxying.
        console.warn(`Unsupported content item type (skipping): "${item.type}"`);
         //throw new HttpError(`Unsupported content item type: "${item.type}"`, 400); // Or skip?
        break;
       case "function":
       case "tool_code_result":
         // These are Gemini-specific input part types, not standard in OpenAI user content array.
         console.warn(`Unsupported content item type (skipping): "${item.type}"`);
         break;
      default:
        console.warn(`Unknown "content" item type (skipping): "${item.type}"`);
        // throw new HttpError(`Unknown "content" item type: "${item.type}"`, 400); // Or skip?
    }
  }

   // Google API sometimes requires a text part in the first user turn if it contains other modalities.
   // Or if it contains *only* images/audio. Add an empty text part if no text was present.
   // This prevents "Unable to submit request because it must have a text parameter" errors for image-only prompts.
  if (parts.length > 0 && !hasText) {
       // Check if any part is actually a text part
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
    // Filter for function tools, as that's the primary supported type via Google tools
    const funcs = req.tools.filter(tool => tool.type === "function");
    if (funcs.length > 0) {
         // Adjust the schemas within the function definitions
         funcs.forEach(adjustSchema);
         // Google API expects an array of tool objects, each containing function_declarations
         tools = [{ function_declarations: funcs.map(tool => tool.function) }];
    } else {
        // If tools array is present but has no functions, send empty tools or null?
        // Sending empty array might be safer.
         tools = [];
    }
  } else if (req.tools !== undefined && req.tools !== null) {
       // If tools is provided but not an array, it's invalid
       console.warn("Request 'tools' field is not an array.", req.tools);
        // Depending on desired behavior, could throw or just ignore invalid field
        // throw new HttpError("'tools' field must be an array.", 400);
        // Ignoring for now allows some leniency
  }


  if (req.tool_choice !== undefined) {
    // Handle tool_choice logic (none, auto, required, or specific function)
    if (typeof req.tool_choice === "string") {
      // 'none', 'auto', 'required'
      const mode = req.tool_choice.toUpperCase();
      if (['NONE', 'AUTO', 'REQUIRED'].includes(mode)) {
         tool_config = { function_calling_config: { mode: mode } };
      } else {
         throw new HttpError(`Unsupported tool_choice string value: "${req.tool_choice}". Must be "none", "auto", or "required".`, 400);
      }
    } else if (typeof req.tool_choice === "object" && req.tool_choice !== null && req.tool_choice.type === "function") {
        // Specific function call: { type: "function", function: { name: "my_function" } }
        const functionName = req.tool_choice.function?.name;
        if (typeof functionName === "string") {
             // Gemini allows specifying allowed function names with mode: ANY
            tool_config = { function_calling_config: { mode: "ANY", allowed_function_names: [functionName] } };
            // Note: OpenAI 'required' type for tool_choice maps to Gemini mode: REQUIRED.
            // A specific function name with OpenAI type 'function' maps to Gemini mode: ANY + allowed_function_names.
            // The original code combined these. Let's keep the specific function -> ANY + allowed_names mapping.
        } else {
             throw new HttpError("Invalid tool_choice object: 'function.name' string is required for type 'function'.", 400);
        }
    } else {
       throw new HttpError("Invalid tool_choice format.", 400);
    }
  }

  return { tools, tool_config };
};


// Combines transformations for the full request body
const transformRequest = async (req) => {
    // Transform messages (handles roles, content parts including images, and system_instruction)
    const { system_instruction, contents } = await transformMessages(req.messages);

    // Transform generation configuration (temperature, max_tokens, etc.)
    const generationConfig = transformConfig(req);

    // Transform tools and tool_choice
    const { tools, tool_config } = transformTools(req);

    // Construct the final Gemini request body
    const geminiRequestBody = {
        contents, // The main conversation turns
        ...(system_instruction && { system_instruction }), // Add system_instruction if present
        generationConfig, // Config settings
        safetySettings, // Default safety settings
        ...(tools && tools.length > 0 && { tools }), // Add tools if present and not empty
        ...(tool_config && { tool_config }), // Add tool_config if present
        // Add other potential top-level fields if needed (e.g., tool_config)
    };

    // console.info("Final Gemini Request Body:", JSON.stringify(geminiRequestBody, null, 2));

    return geminiRequestBody;
};


// --- Response Processing and Streaming ---

// Generate an OpenAI-like ID
const generateId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  // OpenAI IDs are typically prefixed and ~29 random chars long.
  return Array.from({ length: 29 }, randomChar).join("");
};

// Mapping Gemini finishReason to OpenAI finish_reason
const reasonsMap = {
  //"FINISH_REASON_UNSPECIFIED": "stop", // Default? Map to stop? Google says unused.
  "STOP": "stop", // Model finished normally
  "MAX_TOKENS": "length", // Reached max output tokens
  "SAFETY": "content_filter", // Blocked by safety settings
  "RECITATION": "content_filter", // Blocked due to potential recitation of training data
  "OTHER": "other", // Any other reason
  // Add other potential reasons like TOOL_FUNCTION_CALL if not covered by default mapping?
  // If a model finishes with a tool_code or function_call, OpenAI uses "tool_calls" finish reason.
  // We need to check the content parts for this. Handled in transform functions below.
};

// Transforms a single Gemini candidate object (from non-streaming response)
// into an OpenAI-like choice object.
const transformMessageResponse = (cand) => {
  const message = {
      role: "assistant", // Model role is assistant in OpenAI format
      content: [], // Initialize content as an array of parts
      // tool_calls: undefined // Initialize tool_calls (will be added if functionCall parts exist)
  };
  let hasFunctionCall = false; // Flag to check if function calls are present

  // Process content parts from the Gemini candidate
  if (cand.content?.parts && Array.isArray(cand.content.parts)) {
      const tool_calls = []; // Collect function calls for this message
      const content_parts = []; // Collect text and image parts for content array

      for (const part of cand.content.parts) {
          if (part.text !== undefined) {
              // Text part
              content_parts.push({ type: "text", text: part.text });
          } else if (part.inlineData !== undefined) {
              // Inline image data part (base64)
              if (part.inlineData.mimeType && part.inlineData.data) {
                   content_parts.push({
                       type: "image_url",
                       image_url: {
                           url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`
                           // detail: "auto" // Optional, based on OpenAI spec
                       }
                   });
              } else {
                  console.warn("Skipping invalid inlineData part:", part.inlineData);
              }
          } else if (part.functionCall !== undefined) {
              // Function call part
              const fc = part.functionCall;
              if (fc.name && fc.args !== undefined) { // Check for required functionCall fields
                   tool_calls.push({
                       // OpenAI tool_call object structure
                       id: fc.id ?? "call_" + generateId(), // Use Google ID if present, otherwise generate OpenAI-like ID
                       type: "function",
                       function: {
                           name: fc.name,
                           arguments: JSON.stringify(fc.args), // Arguments must be stringified JSON
                       }
                   });
                   hasFunctionCall = true;
              } else {
                   console.warn("Skipping invalid functionCall part:", part.functionCall);
              }
          }
          // Add handling for other part types if needed (e.g., fileData, functionResponse - though functionResponse is input)
      }

      // Assign collected parts to the message object
      message.content = content_parts.length > 0 ? content_parts : null; // Use null if no text/image parts
      if (tool_calls.length > 0) {
          message.tool_calls = tool_calls; // Add tool_calls array if function calls were found
      }
  } else {
      // If no content or parts array, message content is null
      message.content = null;
  }


  // Determine finish reason
  let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || null; // Default to null if mapping fails

  // If the message includes tool_calls, the finish reason is typically "tool_calls" in OpenAI format
  if (hasFunctionCall) {
      finish_reason = "tool_calls";
  }


  // Return the transformed choice object
  return {
    index: cand.index || 0, // Ensure index is present (default to 0 if missing)
    message: message, // The transformed message object
    logprobs: null, // logprobs not typically available/mapped in Gemini
    finish_reason: finish_reason, // The determined finish reason
    // original_finish_reason: cand.finishReason, // Optional: include original for debugging
  };
};

// Transforms a single Gemini candidate object (from streaming chunk)
// into an OpenAI-like delta object for a stream chunk.
const transformDeltaResponse = (cand) => {
    // Delta chunks represent *changes* or additions to the message content.
    // For multimodal content (array of parts), each chunk might add new parts or text to existing parts.
    // Gemini streaming sends fragments, where each chunk's `content.parts` contains
    // the *new* parts for that chunk. The client is responsible for concatenating/appending.

    const delta = {
        // role: "assistant", // Role is typically only in the first delta chunk for a candidate
        content: [], // Initialize delta content as an array of parts
        // tool_calls: undefined // Initialize tool_calls (will be added if functionCall parts exist)
    };
    let hasFunctionCall = false; // Flag to check if function calls are present in this delta chunk

    // Process content parts from the Gemini streaming candidate fragment
     if (cand.content?.parts && Array.isArray(cand.content.parts)) {
        const delta_tool_calls = []; // Collect function calls for this delta chunk
        const delta_content_parts = []; // Collect text and image parts for delta content array

        for (const part of cand.content.parts) {
            if (part.text !== undefined) {
                 // Text part: add it as a text part to the delta content array
                delta_content_parts.push({ type: "text", text: part.text });
            } else if (part.inlineData !== undefined) {
                 // Inline image data part (base64): add it as an image_url part
                 // Note: Images are typically sent as complete parts in a single chunk, not streamed piecemeal.
                 if (part.inlineData.mimeType && part.inlineData.data) {
                     delta_content_parts.push({
                         type: "image_url",
                         image_url: {
                             url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`
                             // detail: "auto" // Optional
                         }
                     });
                 } else {
                     console.warn("Skipping invalid inlineData part in stream chunk:", part.inlineData);
                 }
            } else if (part.functionCall !== undefined) {
                 // Function call part: add it to the delta tool_calls array
                 // Function calls are also typically sent as complete objects in a single chunk.
                 const fc = part.functionCall;
                 if (fc.name && fc.args !== undefined) {
                     delta_tool_calls.push({
                          // OpenAI tool_call object structure in delta
                          // The ID and type might only be in the first chunk for a call, but Gemini often sends the whole object.
                          id: fc.id ?? "call_" + generateId(), // Use Google ID or generate
                          type: "function",
                          function: {
                              name: fc.name,
                              arguments: JSON.stringify(fc.args), // Arguments as stringified JSON
                          }
                     });
                     hasFunctionCall = true; // Flag that this chunk includes function calls
                 } else {
                      console.warn("Skipping invalid functionCall part in stream chunk:", part.functionCall);
                 }
            }
             // Add handling for other part types if needed in delta chunks
        }

        // Assign collected parts to the delta object if they exist
        if (delta_content_parts.length > 0) {
            delta.content = delta_content_parts; // Delta content is an array of parts
        } else {
            delete delta.content; // Remove content field if no text/image parts in this chunk
        }

        if (delta_tool_calls.length > 0) {
             delta.tool_calls = delta_tool_calls; // Delta tool_calls is an array of tool_call objects
        } else {
            delete delta.tool_calls; // Remove tool_calls field if no function calls in this chunk
        }
     }

    // Determine finish reason for this chunk (only present in the final chunk)
    let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || null;

     // If this chunk includes function calls, set finish reason to "tool_calls"
     // This assumes the chunk signaling function calls is the final one or indicates this state.
     // Google's API typically sends the finishReason="STOP" *after* the functionCall parts.
     // The check `if (hasFunctionCall)` should likely be done on the *cumulative* message,
     // but in streaming we only have the delta. Let's rely on Google's `finishReason`.
     // If Google sends `finishReason: "STOP"` *after* sending function calls,
     // mapping STOP to "stop" is correct. OpenAI uses "tool_calls" when the model
     // *stops* to wait for a tool response. If Google indicates this state via finishReason
     // other than STOP (e.g., FUNCTION_CALL), map that. If it's still STOP, we might
     // need to infer "tool_calls" if the *last* chunk contained function calls.
     // Let's stick to mapping Google's `finishReason` for now. If a chunk *has* function calls,
     // maybe override the finish reason for *that specific chunk* if it's not already set?
     // No, the finish reason applies to the *entire* generation process up to that point.
     // Let the mapped `finish_reason` stand if present in the chunk.

    // Return the transformed choice delta object
    return {
        index: cand.index || 0, // Ensure index is present
        delta: delta, // The transformed delta object
        finish_reason: finish_reason, // The mapped finish reason (if present in chunk)
        // logprobs: null, // Not in delta
        // original_finish_reason: cand.finishReason, // Optional
    };
};


// Checks for prompt blocking feedback and adds a content filter choice if blocked
const checkPromptBlock = (choices, promptFeedback, key) => {
  // 'choices' is the array of transformed choice/delta objects
  // 'promptFeedback' is from the Google API response
  // 'key' is either "message" (non-stream) or "delta" (stream)
  if (choices.length > 0) {
      // If choices already exist, prompt was not blocked (or was partially blocked, which Google API handles differently)
      return false;
  }

  // If no choices were generated, check if the prompt was blocked
  if (promptFeedback?.blockReason) {
    console.log("Prompt was blocked by Google API. Reason:", promptFeedback.blockReason);
    // Log safety ratings if available and blocked by safety
    if (promptFeedback.blockReason === "SAFETY" && promptFeedback.safetyRatings) {
      promptFeedback.safetyRatings
        .filter(r => r.blocked) // Log only the safety categories that caused the block
        .forEach(r => console.log(`- Safety Category: ${r.category}, Probability: ${r.probability}, Blocked: ${r.blocked}`));
    }

    // Add a single choice indicating content filtering
    choices.push({
      index: 0, // Always index 0 for the single blocked choice
      [key]: (key === "message") ? null : {}, // 'message' is null, 'delta' is empty object
      finish_reason: "content_filter", // Standard OpenAI reason for content filtering
      // original_finish_reason: promptFeedback.blockReason, // Optional: include original
      // blocked_reason: promptFeedback.blockReason, // Optional: include original block reason
      // blocked_safety_ratings: promptFeedback.safetyRatings, // Optional: include ratings
    });
    return true; // Indicate that the prompt was blocked
  }
  return false; // Indicate that the prompt was not blocked
};


// Processes the full non-streaming response body from Google API
// Transforms it into the OpenAI chat completion JSON string.
const processCompletionsResponse = (data, model, id) => {
   // 'data' is the parsed JSON object from the Google API response body
   // 'model' is the model name requested (e.g., "gemini-pro", "models/gemini-pro")
   // 'id' is the generated OpenAI-like completion ID

  const obj = {
    id, // Generated ID
    choices: [], // Array of transformed choice objects
    created: Math.floor(Date.now()/1000), // Unix timestamp
    model: model.startsWith("models/") ? model : "models/" + model, // Report model name, include "models/" prefix for consistency? Or match request? Let's match request format if possible, or use base name. OpenAI uses base name. Let's use base name.
    // Use the base model name (without "models/")
    model: model.startsWith("models/") ? model.substring(7) : model,
    // system_fingerprint: data.system_fingerprint ?? "fp_unknown", // Include system fingerprint if available (Google API might not provide this)
    object: "chat.completion", // Standard OpenAI object type
    // usage: ... // Usage is added after checking for prompt block
  };

  // Transform Google API candidates into OpenAI choices
  if (data.candidates && Array.isArray(data.candidates)) {
     obj.choices = data.candidates.map(transformMessageResponse);
  } else {
      // If no candidates array, it might be an empty successful response or a block.
       // Handle the case where 'candidates' is missing but response was 200 OK.
       // Google API usually includes `candidates: []` if no generation occurred.
       // If `data.candidates` is genuinely missing, treat as no candidates generated.
       console.warn("Google API response missing 'candidates' array.");
       obj.choices = []; // Ensure choices is an array
  }


  // Check for prompt blocking *after* attempting to transform candidates.
  // If the prompt was blocked, this function will add a content_filter choice.
  checkPromptBlock(obj.choices, data.promptFeedback, "message");


  // Add usage metadata if available from Google API response
  if (data.usageMetadata) {
    obj.usage = transformUsage(data.usageMetadata);
  } else if (data.promptFeedback?.tokenCount) {
      // Sometimes prompt token count is in promptFeedback if generation was blocked
       obj.usage = { prompt_tokens: data.promptFeedback.tokenCount, completion_tokens: 0, total_tokens: data.promptFeedback.tokenCount };
  } else {
      // If no usage info, set to undefined or null
      obj.usage = undefined; // or null;
  }

  // Stringify the final object for the response body
  return JSON.stringify(obj, null, obj.usage ? undefined : "  "); // Pretty print if no usage? Or always compact? Always compact is standard.
  // return JSON.stringify(obj); // Standard compact JSON output
};

// Transforms Google API usageMetadata to OpenAI usage object
const transformUsage = (data) => {
  // Mapping Google fields to OpenAI fields
  const usage = {
    prompt_tokens: data.promptTokenCount ?? 0, // Tokens in the prompt
    completion_tokens: data.candidatesTokenCount ?? 0, // Tokens in the generated response(s)
    total_tokens: data.totalTokenCount ?? 0, // Total tokens (prompt + completion)
    // Add other fields if available and desired:
    // total_billable_characters: data.totalBillableCharacters,
  };
   // Ensure fields are numbers
   Object.keys(usage).forEach(key => { usage[key] = Number(usage[key]) || 0; });
   return usage;
};


// --- Streaming Transform Functions ---

// Regex to parse Server-Sent Events 'data:' lines
// It captures the content after "data: " and before the double newline (\n\n, \r\r, or \r\n\r\n)
const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/s; // Added 's' flag for multiline match if needed, though SSE lines are usually single.
// Let's remove 's' as it's not standard for SSE lines and might cause unexpected behavior.
const responseLineRE_single = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;

// TransformStream for parsing SSE data chunks.
// Accumulates buffer and emits data lines as they are complete.
function parseStream (chunk, controller) {
  this.buffer += chunk; // Append new chunk to buffer
  // Process complete lines from the buffer
  let match;
  while ((match = this.buffer.match(responseLineRE_single)) !== null) {
    // Enqueue the captured data payload (the content after "data: ")
    controller.enqueue(match[1]);
    // Remove the processed line from the buffer
    this.buffer = this.buffer.buffer.slice(match[0].length).toString(); // Use buffer.slice and toString for efficiency
  }
}

// Flush function for parseStream. Handles any remaining data in the buffer.
function parseStreamFlush (controller) {
  if (this.buffer.length > 0) {
    console.error("parseStream: Remaining data in buffer after flush:", this.buffer);
    // Decide how to handle leftover data: enqueue it? throw error?
    // Enqueueing might pass incomplete JSON, which the next transform needs to handle.
    // Let's enqueue it but set a flag in shared state.
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true; // Indicate leftover data was sent
  }
  // Reset buffer for the next stream if this were part of a larger pipeline (not typically needed for the final stream)
  this.buffer = "";
}

// Delimiter for SSE messages
const delimiter = "\n\n";

// Helper to format an object into an SSE 'data:' line
const sseline = (obj) => {
  // Ensure the object is stringified JSON
  const dataString = JSON.stringify(obj);
  return `data: ${dataString}${delimiter}`;
};

// TransformStream for transforming Google API stream chunks (parsed JSON)
// into OpenAI API stream chunks (SSE 'data:' lines).
function toOpenAiStream (line, controller) {
  // 'line' is a JSON string emitted from parseStream
  let data;
  try {
    data = JSON.parse(line);
    // Check if the parsed data looks like a valid Google API stream chunk fragment
    if (data.error) {
        // Handle errors embedded in the stream (Google API might send {error: ...} even in 200 stream)
        console.error("Google API stream error chunk:", data.error);
        // Transform the error into an OpenAI-like error chunk? Or stop stream?
        // For now, let's create an error chunk and signal done.
        const errorChunk = {
            id: this.id,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now()/1000),
            model: this.model.startsWith("models/") ? this.model.substring(7) : this.model,
            choices: [{
                index: 0,
                delta: { role: "assistant" }, // First chunk typically has role
                finish_reason: "tool_error", // Or map Google error reason?
            }],
            error: { // Add an error field similar to OpenAI error format (though not standard in chunk)
                message: data.error.message || "An error occurred during streaming.",
                type: data.error.status || "api_error",
                code: data.error.code,
             },
        };
        controller.enqueue(sseline(errorChunk));
        // Signal end of stream after an error chunk
        controller.enqueue("data: [DONE]" + delimiter);
        return; // Stop processing this chunk
    }
     // A valid chunk fragment should typically have 'candidates' or 'promptFeedback'
    if (!data.candidates && !data.promptFeedback) {
       // This might be unexpected data.
       if (this.shared.is_buffers_rest && line.trim().length > 0) {
           // If it's leftover buffer data from parsing, maybe log and ignore?
           console.warn("toOpenAiStream: Ignoring potentially incomplete or malformed chunk:", line);
       } else {
           console.warn("toOpenAiStream: Unexpected Google API stream chunk format (missing candidates/promptFeedback):", data);
           // Decide whether to enqueue as is, throw, or skip. Skipping might drop data.
           // Enqueueing raw might break client JSON parsing. Throwing stops the stream.
           // Let's log and skip for now, assuming it's non-critical metadata or malformed data.
           // Consider throwing in production if unexpected chunks are common.
           // throw new Error("Unexpected Google API stream chunk format.");
       }
       return; // Skip processing this chunk
    }


  } catch (err) {
    // Handle JSON parsing errors from the input line
    console.error("toOpenAiStream: Error parsing JSON line:", err, "Line:", line);
     // If the line wasn't just leftover buffer data
    if (!this.shared.is_buffers_rest || line.trim().length > 0) {
       // Decide how to handle parsing error: enqueue original line? throw?
       // Enqueueing original line might break client SSE parsing if it's not 'data: ...' format.
       // Throwing stops the stream.
       // Let's throw to signal a critical error in the stream format.
        controller.error(new Error("Failed to parse stream chunk JSON: " + err.message));
       // Or enqueue an error chunk and DONE:
       // controller.enqueue(sseline({ id: this.id, object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "tool_error" }], error: { message: "Failed to parse stream chunk JSON.", type: "parse_error" } }));
       // controller.enqueue("data: [DONE]" + delimiter);
       // return;
    }
    return; // Skip if it was just leftover buffer
  }

  // Handle prompt blocking feedback received in a chunk
  // This chunk will likely have promptFeedback but no candidates.
  if (data.promptFeedback) {
      const choices = []; // Array to hold the single content_filter choice
      if (checkPromptBlock(choices, data.promptFeedback, "delta")) {
          // checkPromptBlock adds the { index: 0, delta: {}, finish_reason: "content_filter" } choice
          const blockChunk = {
             id: this.id,
             object: "chat.completion.chunk",
             created: Math.floor(Date.now()/1000),
             model: this.model.startsWith("models/") ? this.model.substring(7) : this.model, // Report model name
             choices: choices, // Contains the single blocked choice
             // Usage might be included here by Google API in promptFeedback
             usage: data.promptFeedback.tokenCount ? { prompt_tokens: data.promptFeedback.tokenCount, completion_tokens: 0, total_tokens: data.promptFeedback.tokenCount } : undefined,
          };
           controller.enqueue(sseline(blockChunk));
           // A prompt block effectively ends the generation for this prompt
           controller.enqueue("data: [DONE]" + delimiter);
           return; // Stop processing this chunk
      }
       // If promptFeedback is present but not a block (e.g., just usage info), continue processing candidates below
       if (data.candidates === undefined || data.candidates === null) {
            // If a chunk only had promptFeedback and no candidates, and wasn't a block,
            // maybe it's just usage info or metadata? Skip processing candidates part.
            // However, Google's stream usageMetadata is often sent in the final chunk *with* the last candidate data.
            // If a chunk has promptFeedback but no candidates, and checkPromptBlock didn't trigger,
            // it's likely just metadata or an unexpected format. Let's skip.
            console.warn("toOpenAiStream: Chunk contains promptFeedback but no candidates and is not a block.", data);
            return;
       }
  }


  // Process candidates (should be an array, typically with 1 candidate in streaming)
  if (data.candidates && Array.isArray(data.candidates)) {
    // Transform each candidate's delta into OpenAI format
    const openaiChoices = data.candidates.map(transformDeltaResponse);

    // Add role="assistant" to the first delta chunk for candidate 0
    // Store state for each candidate index using this.last
    openaiChoices.forEach(choice => {
        const index = choice.index || 0;
        if (!this.last[index]) {
            // This is the first chunk for this candidate index
            this.last[index] = { delta: {}, finish_reason: null }; // Initialize state for this candidate
            // Add role to the first delta chunk for index 0
            if (index === 0) {
                 choice.delta = choice.delta || {};
                 choice.delta.role = "assistant";
            }
        }
        // Update accumulated state (finish reason might appear in later chunks)
        if (choice.finish_reason) {
            this.last[index].finish_reason = choice.finish_reason;
        }

        // Enqueue the transformed OpenAI chunk
        const openaiChunk = {
           id: this.id,
           object: "chat.completion.chunk",
           created: Math.floor(Date.now()/1000),
           model: this.model.startsWith("models/") ? this.model.substring(7) : this.model, // Report model name
           choices: [choice], // Each chunk contains one choice object in OpenAI format
           // Usage metadata is typically in the final chunk. Add it here if requested and available.
           usage: data.usageMetadata && this.streamIncludeUsage ? transformUsage(data.usageMetadata) : undefined,
        };

         // Only enqueue if the delta has content or tool_calls, or if it's the first chunk (to include role), or if it has a finish reason or usage.
         // An empty delta ({}) with no finish_reason or usage is just noise.
         const deltaHasContent = choice.delta && Object.keys(choice.delta).length > 0;
         const isFirstChunkForIndex0 = index === 0 && choice.delta?.role === "assistant"; // Check if it's the very first chunk for candidate 0
         const chunkHasFinishReasonOrUsage = choice.finish_reason || openaiChunk.usage;

         if (deltaHasContent || isFirstChunkForIndex0 || chunkHasFinishReasonOrUsage) {
             controller.enqueue(sseline(openaiChunk));
         } else {
             // console.log(`Skipping empty chunk for index ${index}:`, openaiChunk);
         }

    });

  } else {
     // If data.candidates is not an array and it wasn't a prompt block,
     // it's an unexpected structure. Log and potentially skip or error.
      console.warn("toOpenAiStream: Chunk missing 'candidates' array and not a prompt block:", data);
      // Decide whether to throw or skip. Skipping is more resilient but hides issues.
      // throw new Error("Invalid stream chunk format: missing 'candidates'.");
      return; // Skip processing this chunk
  }
}

// Flush function for toOpenAiStream. Sends the final [DONE] signal.
function toOpenAiStreamFlush (controller) {
  // After processing all chunks, send the final [DONE] signal.
  // We might also send final usage metadata here if it wasn't in the last chunk and include_usage was true.
  // Google often puts final usage in the last data chunk. If not, or if include_usage is true,
  // we need to ensure usage is sent. The current logic adds usage if `data.usageMetadata` is in the chunk.
  // If the *very last* chunk received had usageMetadata, it was already added.
  // If it wasn't in the last chunk but is conceptually final, Google API might send it in a standalone chunk
  // which would be processed above. So, relying on `data.usageMetadata` in the stream chunks seems sufficient.


  // Send the final DONE signal
  controller.enqueue("data: [DONE]" + delimiter);

  // Clean up state
  this.buffer = ""; // Not used in this stream, but good practice
  this.last = [];
  this.shared = {};
}

// The SEP constant is no longer used with array content, can be removed.
// const SEP = "\n\n|>"; // Remove this line

// --- END OF RESPONSE PROCESSING AND STREAMING ---
