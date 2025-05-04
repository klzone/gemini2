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
      // Include Google API error details if available in the original error
       let message = err.message;
       if (err.originalResponseText) {
           try {
               const originalError = JSON.parse(err.originalResponseText);
               if (originalError.error?.message) {
                   // Prefer Google's detailed error message if present
                   message = originalError.error.message;
               } else if (originalError.message) {
                   // Fallback to a non-standard top-level message
                    message = originalError.message;
               }
           } catch (parseErr) {
               // If original response text isn't JSON, just use the current message
               console.error("Failed to parse original response text for error details:", parseErr);
           }
       }


      // Structure the error response body similar to OpenAI's error object
      const errorBody = {
           error: {
               message: message || "An unknown error occurred.",
               type: (err instanceof HttpError) ? "api_error" : "internal_error", // Categorize errors
               code: status, // Use the HTTP status code as the error code
               // Include original Google API status if it's an HttpError that wrapped a response
               ...(err.originalResponseStatus && { original_status: err.originalResponseStatus }),
               // Add other details if needed, e.g., specific Google error codes
           }
      };

      return new Response(JSON.stringify(errorBody), fixCors({ status }));
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
      // This makes the endsWith checks work regardless of whether the client
      // includes /v1 in the base URL or the path.
      const normalizedPathname = pathname.startsWith('/v1') ? pathname.substring(3) : pathname;


      switch (true) {
        // Handles chat models (including multimodal like image generation through chat)
        case normalizedPathname.endsWith("/chat/completions"):
          assert(request.method === "POST", "Method Not Allowed", 405);
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);

        // Handles embeddings models
        case normalizedPathname.endsWith("/embeddings"):
          assert(request.method === "POST", "Method Not Allowed", 405);
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);

        // Handles listing models
        case normalizedPathname.endsWith("/models"):
          assert(request.method === "GET", "Method Not Allowed", 405);
          return handleModels(apiKey)
            .catch(errHandler);

        // Handles OpenAI-style image generation endpoint
        case normalizedPathname.endsWith("/images/generations"):
          assert(request.method === "POST", "Method Not Allowed", 405);
          return handleImageGenerations(await request.json(), apiKey)
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
  constructor(message, status, originalResponseText = null, originalResponseStatus = null) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
    this.originalResponseText = originalResponseText; // Store original response body text if available
    this.originalResponseStatus = originalResponseStatus; // Store original response status if available
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "*"); // Allow all methods for simplicity
  headers.set("Access-Control-Allow-Headers", "*"); // Allow all headers for simplicity
  headers.set("Content-Type", "application/json"); // Assume JSON response for most cases

  // Add specific headers if needed
  // headers.set("Cache-Control", "no-cache, must-revalidate");

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
     // Wrap fetch error in HttpError
    throw new HttpError("Failed to read response body from Google API: " + e.message, 500);
  }

  if (response.ok) {
    try {
      const data = JSON.parse(responseBodyText);
       if (!data.models || !Array.isArray(data.models)) {
            // Check if it's an error structure despite 200 status
            if (data.error) {
               console.error("Google API error in Models JSON body:", data.error);
               // Re-throw as HttpError to be caught by errHandler
                throw new HttpError(data.error.message || "Google API error listing models.", data.error.code || 500, responseBodyText, response.status);
            }
            throw new Error("Unexpected response format: 'models' array not found or invalid.");
       }

      body = JSON.stringify({
        object: "list",
        data: data.models.map(({ name }) => ({
          id: name.replace("models/", ""),
          object: "model",
          created: 0, // Static, as original API doesn't provide timestamp
          owned_by: "google",
        })),
      }, null, "  "); // Pretty print for debugging
    } catch (err) {
      console.error("Error processing models response:", err);
       // If transformation fails, wrap in HttpError
       // If err is already HttpError (from recursive check), re-throw it
       if (err instanceof HttpError) throw err;
      throw new HttpError("Error processing models response from Google API: " + err.message, 500, responseBodyText, response.status);
    }
  } else {
     // If the original response was not ok, wrap the error details
      console.error("Google API returned error status for models:", response.status, response.statusText, responseBodyText);
      // Try to parse for a more specific message, otherwise use status text
      let errorMessage = response.statusText;
      try {
          const errorData = JSON.parse(responseBodyText);
          if (errorData.error?.message) {
              errorMessage = errorData.error.message;
          } else if (errorData.message) {
              errorMessage = errorData.message;
          }
      } catch (parseErr) {
          console.error("Failed to parse Google API error body for models:", parseErr);
          // Use default statusText if parsing fails
      }
       throw new HttpError(errorMessage, response.status, responseBodyText, response.status);
  }

  // Return the transformed body or the original error body with CORS headers
  return new Response(body, fixCors(response));
}


const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKey) {
  // req is the parsed JSON request body from the client

  let modelName = req.model; // Get the model name from the request

  if (typeof modelName !== "string" || !modelName) {
     console.warn(`Model not specified or invalid for embeddings. Using default: ${DEFAULT_EMBEDDINGS_MODEL}`);
     modelName = DEFAULT_EMBEDDINGS_MODEL;
  }

   // Ensure the model name has the 'models/' prefix for the Google API endpoint
  const modelEndpoint = modelName.startsWith("models/") ? modelName : "models/" + modelName;

  // Input can be a string or array of strings
  if (!Array.isArray(req.input)) {
     if (req.input === null || req.input === undefined) {
         throw new HttpError("Input is required for embeddings.", 400);
     }
    req.input = [ String(req.input) ]; // Ensure array of strings
  } else {
     req.input = req.input.map(item => String(item)); // Ensure all items are strings
      if (req.input.length === 0) {
           throw new HttpError("Input array is empty for embeddings.", 400);
      }
  }


  // API endpoint is batchEmbedContents for multiple inputs
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${modelEndpoint}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        // model needs to be included in each request for batch endpoint
        model: modelEndpoint,
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
        throw new HttpError("Failed to read response body from Google API: " + e.message, 500);
    }


  if (response.ok) {
    try {
      const data = JSON.parse(responseBodyText);
       if (!data.embeddings || !Array.isArray(data.embeddings)) {
            if (data.error) {
                console.error("Google API error in Embeddings JSON body:", data.error);
                 throw new HttpError(data.error.message || "Google API error during embeddings.", data.error.code || 500, responseBodyText, response.status);
            }
           throw new Error("Unexpected response format: 'embeddings' array not found or invalid.");
       }

      body = JSON.stringify({
        object: "list",
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
        if (err instanceof HttpError) throw err;
        throw new HttpError("Error processing embeddings response from Google API: " + err.message, 500, responseBodyText, response.status);
    }
  } else {
     console.error("Google API returned error status for embeddings:", response.status, response.statusText, responseBodyText);
     let errorMessage = response.statusText;
     try {
         const errorData = JSON.parse(responseBodyText);
         if (errorData.error?.message) {
             errorMessage = errorData.error.message;
         } else if (errorData.message) {
             errorMessage = errorData.message;
         }
     } catch (parseErr) {
         console.error("Failed to parse Google API error body for embeddings:", parseErr);
     }
      throw new HttpError(errorMessage, response.status, responseBodyText, response.status);
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
          modelName = req.model;
      }
  }

  let body = await transformRequest(req); // Transforms OpenAI -> Gemini request body

  // Handle specific model suffix requests like "-search-preview" or ":search"
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
         throw new HttpError("Failed to read response body from Google API: " + e.message, 500);
      }


      try {
        const data = JSON.parse(responseBodyText);
        // Check for expected structure or API errors embedded in JSON
        if (data.error) {
           console.error("Google API error in Chat Completions JSON body:", data.error);
           throw new HttpError(data.error.message || "Google API error during chat completion.", data.error.code || 500, responseBodyText, response.status);

        } else if (!data.candidates && !data.promptFeedback) {
          throw new Error("Invalid completion object: missing 'candidates' or 'promptFeedback'.");
        } else {
           responseBody = processCompletionsResponse(data, req.model, id);
        }
      } catch (err) {
        console.error("Error parsing or processing non-streaming response:", err);
        if (err instanceof HttpError) throw err;
        throw new HttpError("Error processing chat completions response from Google API: " + err.message, 500, responseBodyText, response.status);
      }
    }
  } else {
     console.error("Google API returned error status for chat completions:", response.status, response.statusText, responseBodyText);
     let errorMessage = response.statusText;
     try {
         const errorData = JSON.parse(responseBodyText);
         if (errorData.error?.message) {
             errorMessage = errorData.error.message;
         } else if (errorData.message) {
             errorMessage = errorData.message;
         }
     } catch (parseErr) {
         console.error("Failed to parse Google API error body for chat completions:", parseErr);
     }
      throw new HttpError(errorMessage, response.status, responseBodyText, response.status);
  }

  return new Response(responseBody, fixCors(response));
}


// --- NEW FUNCTION FOR IMAGE GENERATION ---

async function handleImageGenerations(req, apiKey) {
    // This function handles requests to the /images/generations endpoint
    // It expects an OpenAI-like request body for image generation:
    // { prompt: string, n?: number, size?: string, response_format?: { type: "url" | "b64_json" } }
    // It will call the Gemini generateContent API with an image generation model.

    const prompt = req.prompt;
    if (typeof prompt !== 'string' || !prompt || prompt.trim().length === 0) {
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
        safetySettings: safetySettings, // Apply default safety settings
        generationConfig: {
           // OpenAI n -> Gemini candidateCount. Limit to 4 as Gemini typically supports up to 4 candidates.
           candidateCount: req.n !== undefined && typeof req.n === 'number' && req.n >= 1 ? Math.min(Math.floor(req.n), 4) : 1, // Limit n to 1-4, default 1
           // Gemini generateContent for image models doesn't have a 'size' parameter. Ignore req.size.
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
        throw new HttpError("Failed to read response body from Google API: " + e.message, 500);
    }


    if (response.ok) {
        try {
            const data = JSON.parse(responseBodyText);

            // Check for Google API errors embedded in JSON
            if (data.error) {
               console.error("Google API error in Image Generations JSON body:", data.error);
               throw new HttpError(data.error.message || "Google API error during image generation.", data.error.code || 500, responseBodyText, response.status);
            }

             // Handle prompt blocking first
            const choices = [];
            if (checkPromptBlock(choices, data.promptFeedback, "message")) {
                 // If prompt was blocked, checkPromptBlock added a choice with finish_reason: "content_filter"
                 // For image generation, translate this to an OpenAI-style error response structure.
                 let errorMessage = data.promptFeedback.blockReason ? `Prompt blocked: ${data.promptFeedback.blockReason}` : "Prompt blocked.";
                 if (data.promptFeedback.safetyRatings) {
                     errorMessage += " Details: " + data.promptFeedback.safetyRatings
                         .filter(r => r.blocked)
                         .map(r => `${r.category} (${r.probability})`)
                         .join(", ");
                 }
                 // Throw HttpError which errHandler will format
                 throw new HttpError(errorMessage, 400, responseBodyText, response.status);

            }

            // If not blocked, process candidates (should contain inlineData parts for images)
            const imageUrls = []; // Array to hold the image objects ({ url: "..." } or { b64_json: "..." })
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

            if (imageUrls.length === 0 && (!data.candidates || data.candidates.length === 0)) {
                 // If API returned OK but no candidates with images were found, and it wasn't blocked
                 console.warn("Google API returned OK for image generation but found no candidates with image parts.");
                 // Return an empty data array as success, similar to OpenAI behavior for no results.
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
            if (err instanceof HttpError) throw err;
            throw new HttpError("Error processing image generation response from Google API: " + err.message, 500, responseBodyText, response.status);
        }
    } else {
       console.error("Google API returned error status for image generation:", response.status, response.statusText, responseBodyText);
        let errorMessage = response.statusText;
        try {
            const errorData = JSON.parse(responseBodyText);
            if (errorData.error?.message) {
                errorMessage = errorData.error.message;
            } else if (errorData.message) {
                errorMessage = errorData.message;
            }
        } catch (parseErr) {
            console.error("Failed to parse Google API error body for image generation:", parseErr);
        }
       throw new HttpError(errorMessage, response.status, responseBodyText, response.status);
    }

    // Return the final Response object with appropriate headers
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
    if (schemaPart.type === "object" && schemaPart.properties && schemaPart.additionalProperties === false) {
      delete schemaPart.additionalProperties;
    }
    Object.values(schemaPart).forEach(adjustProps);
  }
};
const adjustSchema = (tool) => {
  if (!tool || tool.type !== "function" || !tool.function || !tool.function.parameters) {
      return;
  }
  const parameters = tool.function.parameters;
  if (parameters.type === "object" && parameters.strict !== undefined) {
     delete parameters.strict;
  }
  adjustProps(parameters);
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
  threshold: "BLOCK_NONE",
}));

const fieldsMap = {
  frequency_penalty: "frequencyPenalty",
  max_tokens: "maxOutputTokens",
  presence_penalty: "presencePenalty",
  seed: "seed",
  top_k: "topK",
  top_p: "topP",
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
  if (req.n !== undefined && !req.stream) {
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
        name,
        args,
      }
    };
  });
  parts.callsMapping = callsMapping;
  return parts;
};

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
         if (typeof message.content === 'string' && message.content.trim().length > 0) {
             system_instruction_parts.push({ text: message.content });
         } else {
             console.warn("Skipping empty or invalid system message content.");
         }
        continue;

      case "tool": {
        if (!message.tool_call_id) {
             throw new HttpError("tool_call_id is required for messages with role 'tool'.", 400);
        }
        const functionName = lastAssistantToolCallsMapping[message.tool_call_id];
        if (!functionName) {
             throw new HttpError(`Function name not found for tool_call_id "${message.tool_call_id}". Ensure the immediately preceding assistant message contained this tool_call_id.`, 400);
        }

        let responseData;
        try {
            responseData = JSON.parse(message.content);
        } catch (err) {
             console.error("Error parsing tool message content:", err);
             throw new HttpError("Invalid content in tool message (not valid JSON).", 400);
        }

        contents.push({
            role: "user", // Tool response is part of the user's turn
            parts: [{
                 functionResponse: {
                    name: functionName,
                    response: responseData,
                 }
            }]
        });

        lastAssistantToolCallsMapping = {}; // Reset mapping after a user turn (or tool turn responding for user)
        break;

      }

      case "assistant":
        message.role = "model";

        if (message.tool_calls && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
             const toolCallParts = transformFnCalls(message);
             message.parts = toolCallParts;
             lastAssistantToolCallsMapping = toolCallParts.callsMapping || {};

             if (message.content !== undefined && message.content !== null && message.content !== "") {
                 console.warn("Assistant message has both tool_calls and non-empty content. Prioritizing tool_calls and ignoring content.");
             }
             delete message.content;

        } else {
             message.parts = await transformMsgContent(message);
             lastAssistantToolCallsMapping = {};
             delete message.content;
        }
        contents.push({
            role: message.role,
            parts: message.parts
        });
        break;

      case "user":
        message.role = "user";

        message.parts = await transformMsgContent(message);

         if (!message.parts || message.parts.length === 0) {
             message.parts = [{ text: "" }];
             console.warn("Empty user message content received, adding empty text part.");
         }

         delete message.content;
        lastAssistantToolCallsMapping = {};

        contents.push({
            role: message.role,
            parts: message.parts
        });
        break;


      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
  }

  let system_instruction = system_instruction_parts.length > 0 ? { parts: system_instruction_parts } : undefined;


  if (contents.length > 0 && contents[0].role !== "user") {
      contents.unshift({ role: "user", parts: [{ text: "" }] });
  }

  contents.forEach(content => {
      if (content.parts && Array.isArray(content.parts)) {
           delete content.parts.callsMapping;
      }
  });

  // console.info("Transformed Gemini Request Contents:", JSON.stringify(contents, null, 2));
  // console.info("Transformed Gemini Request System Instruction:", JSON.stringify(system_instruction, null, 2));

  return { system_instruction, contents };
};

const transformMsgContent = async (message) => {
  const parts = [];
  const content = message.content;

  if (content === null || content === undefined || (Array.isArray(content) && content.length === 0)) {
      return parts;
  }

  if (!Array.isArray(content)) {
    parts.push({ text: String(content) });
    return parts;
  }

  let hasText = false;
  for (const item of content) {
    if (typeof item !== 'object' || item === null) {
        console.warn("Unexpected item type in message content array:", item);
        continue;
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
      default:
        console.warn(`Unknown or unsupported "content" item type (skipping): "${item.type}"`);
    }
  }

  if (parts.length > 0 && !hasText) {
       const hasExistingTextPart = parts.some(p => p.text !== undefined);
       if (!hasExistingTextPart) {
           parts.push({ text: "" });
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
         tools = [];
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


const transformRequest = async (req) => {
    const { system_instruction, contents } = await transformMessages(req.messages);
    const generationConfig = transformConfig(req);
    const { tools, tool_config } = transformTools(req);

    const geminiRequestBody = {
        contents,
        ...(system_instruction && { system_instruction }),
        generationConfig,
        safetySettings,
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
};

const transformMessageResponse = (cand) => {
  const message = {
      role: "assistant",
      content: [],
  };
  let hasFunctionCall = false;

  if (cand.content?.parts && Array.isArray(cand.content.parts)) {
      const tool_calls = [];
      const content_parts = [];

      for (const part of cand.content.parts) {
          if (part.text !== undefined) {
              content_parts.push({ type: "text", text: part.text });
          } else if (part.inlineData !== undefined) {
               if (part.inlineData.mimeType && part.inlineData.data) {
                    content_parts.push({
                        type: "image_url",
                        image_url: {
                            url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`
                        }
                    });
               } else {
                   console.warn("Skipping invalid inlineData part in candidate:", part.inlineData);
               }
          } else if (part.functionCall !== undefined) {
              const fc = part.functionCall;
              if (fc.name && fc.args !== undefined) {
                   tool_calls.push({
                       id: fc.id ?? "call_" + generateId(),
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
      }

      message.content = content_parts.length > 0 ? content_parts : null;
      if (tool_calls.length > 0) {
          message.tool_calls = tool_calls;
      }
  } else {
       message.content = null;
  }

  let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || null;

  if (hasFunctionCall) {
      finish_reason = "tool_calls";
  }

  return {
    index: cand.index || 0,
    message: message,
    logprobs: null,
    finish_reason: finish_reason,
  };
};

const transformDeltaResponse = (cand) => {
    const delta = {}; // Start with an empty delta

     if (cand.content?.parts && Array.isArray(cand.content.parts)) {
        const delta_tool_calls = [];
        const delta_content_parts = [];

        for (const part of cand.content.parts) {
            if (part.text !== undefined) {
                 delta_content_parts.push({ type: "text", text: part.text });
            } else if (part.inlineData !== undefined) {
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
                 } else {
                      console.warn("Skipping invalid functionCall part in stream chunk:", part.functionCall);
                 }
            }
        }

        if (delta_content_parts.length > 0) {
            delta.content = delta_content_parts;
        }

        if (delta_tool_calls.length > 0) {
             delta.tool_calls = delta_tool_calls;
        }
     }

    let finish_reason = reasonsMap[cand.finishReason] || cand.finishReason || null;

    return {
        index: cand.index || 0,
        delta: delta,
        finish_reason: finish_reason,
    };
};


const checkPromptBlock = (choices, promptFeedback, key) => {
  if (choices.length > 0) {
      return false;
  }

  if (promptFeedback?.blockReason) {
    console.log("Prompt was blocked by Google API. Reason:", promptFeedback.blockReason);
    if (promptFeedback.blockReason === "SAFETY" && promptFeedback.safetyRatings) {
      promptFeedback.safetyRatings
        .filter(r => r.blocked)
        .forEach(r => console.log(`- Safety Category: ${r.category}, Probability: ${r.probability}, Blocked: ${r.blocked}`));
    }

    choices.push({
      index: 0,
      [key]: (key === "message") ? null : {},
      finish_reason: "content_filter",
    });
    return true;
  }
  return false;
};


const processCompletionsResponse = (data, model, id) => {
  const obj = {
    id,
    choices: [],
    created: Math.floor(Date.now()/1000),
    model: model.startsWith("models/") ? model.substring(7) : model,
    object: "chat.completion",
  };

  if (data.candidates && Array.isArray(data.candidates)) {
     obj.choices = data.candidates.map(transformMessageResponse);
  } else {
      console.warn("Google API response missing 'candidates' array for non-streaming chat completion.");
      obj.choices = [];
  }

  checkPromptBlock(obj.choices, data.promptFeedback, "message");

  if (data.usageMetadata) {
    obj.usage = transformUsage(data.usageMetadata);
  } else if (data.promptFeedback?.tokenCount) {
       obj.usage = { prompt_tokens: data.promptFeedback.tokenCount, completion_tokens: 0, total_tokens: data.promptFeedback.tokenCount };
  } else {
      obj.usage = undefined;
  }

  return JSON.stringify(obj);
};

const transformUsage = (data) => {
  const usage = {
    prompt_tokens: data.promptTokenCount ?? 0,
    completion_tokens: data.candidatesTokenCount ?? 0,
    total_tokens: data.totalTokenCount ?? 0,
  };
   Object.keys(usage).forEach(key => { usage[key] = Number(usage[key]) || 0; });
   return usage;
};


const responseLineRE_single = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;

function parseStream (chunk, controller) {
  this.buffer += chunk;
  let match;
  while ((match = this.buffer.match(responseLineRE_single)) !== null) {
    controller.enqueue(match[1]);
    this.buffer = this.buffer.buffer.slice(match[0].length).toString();
  }
}

function parseStreamFlush (controller) {
  if (this.buffer.length > 0) {
    console.error("parseStream: Remaining data in buffer after flush:", this.buffer);
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true;
  }
  this.buffer = "";
}

const delimiter = "\n\n";

const sseline = (obj) => {
  const dataString = JSON.stringify(obj);
  return `data: ${dataString}${delimiter}`;
};

function toOpenAiStream (line, controller) {
  let data;
  try {
    data = JSON.parse(line);
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
                finish_reason: "tool_error",
            }],
            error: {
                message: data.error.message || "An error occurred during streaming.",
                type: data.error.status || "api_error",
                code: data.error.code,
             },
        };
        controller.enqueue(sseline(errorChunk));
        controller.enqueue("data: [DONE]" + delimiter);
        return;
    }
    if (!data.candidates && !data.promptFeedback) {
       if (this.shared.is_buffers_rest && line.trim().length > 0) {
           console.warn("toOpenAiStream: Ignoring potentially incomplete or malformed chunk:", line);
       } else {
           console.warn("toOpenAiStream: Unexpected Google API stream chunk format:", data);
       }
       return;
    }

  } catch (err) {
    console.error("toOpenAiStream: Error parsing JSON line:", err, "Line:", line);
    if (!this.shared.is_buffers_rest || line.trim().length > 0) {
        controller.error(new Error("Failed to parse stream chunk JSON: " + err.message));
    }
    return;
  }

  if (data.promptFeedback) {
      const choices = [];
      if (checkPromptBlock(choices, data.promptFeedback, "delta")) {
          const blockChunk = {
             id: this.id,
             object: "chat.completion.chunk",
             created: Math.floor(Date.now()/1000),
             model: this.model.startsWith("models/") ? this.model.substring(7) : this.model,
             choices: choices,
             usage: data.promptFeedback.tokenCount ? { prompt_tokens: data.promptFeedback.tokenCount, completion_tokens: 0, total_tokens: data.promptFeedback.tokenCount } : undefined,
          };
           controller.enqueue(sseline(blockChunk));
           controller.enqueue("data: [DONE]" + delimiter);
           return;
      }
       if (!data.candidates) {
           console.warn("toOpenAiStream: Chunk contains promptFeedback but no candidates and is not a block.", data);
           return;
       }
  }

  if (data.candidates && Array.isArray(data.candidates)) {
    const openaiChoices = data.candidates.map(transformDeltaResponse);

    openaiChoices.forEach(choice => {
        const index = choice.index || 0;
        if (!this.last[index]) {
            this.last[index] = { delta: {}, finish_reason: null };
        }

        if (index === 0 && Object.keys(this.last[index].delta).length === 0 && !choice.delta.role) {
             choice.delta = choice.delta || {};
             choice.delta.role = "assistant";
        }

        if (choice.finish_reason) {
            this.last[index].finish_reason = choice.finish_reason;
        }

        const openaiChunk = {
           id: this.id,
           object: "chat.completion.chunk",
           created: Math.floor(Date.now()/1000),
           model: this.model.startsWith("models/") ? this.model.substring(7) : this.model,
           choices: [choice],
           usage: data.usageMetadata && this.streamIncludeUsage ? transformUsage(data.usageMetadata) : undefined,
        };

         const deltaHasMeaningfulContent = choice.delta && (Object.keys(choice.delta).length > 0 || choice.delta.role);
         const chunkHasFinishReasonOrUsage = choice.finish_reason || openaiChunk.usage;

         if (deltaHasMeaningfulContent || chunkHasFinishReasonOrUsage) {
             controller.enqueue(sseline(openaiChunk));
         }
    });

  } else {
      console.warn("toOpenAiStream: Chunk missing 'candidates' array and not a prompt block:", data);
  }
}

function toOpenAiStreamFlush (controller) {
  controller.enqueue("data: [DONE]" + delimiter);
  this.last = [];
  this.shared = {};
}
