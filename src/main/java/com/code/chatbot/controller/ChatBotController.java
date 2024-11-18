package com.code.chatbot.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.document.Document;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/chatbot")
public class ChatBotController {

    private final VectorStore vectorStore;
    @Value("classpath:/prompts/rag-prompt-template.st")
    private Resource ragPromptTemplate;

    public ChatBotController(ChatClient.Builder chatClientBuilder, VectorStore vectorStore) {
        chatClientBuilder.build();
        this.vectorStore = vectorStore;
    }


    @GetMapping("/rag")
    public Generation chatbot(@RequestParam(value = "message", defaultValue = "Lợi ích của RAG") String message) {
        List<Document> similarDocument = vectorStore.similaritySearch(SearchRequest.query(message).withTopK(2));
        List<String> contentList = similarDocument.stream().map(Document::getContent).toList();

        // Đọc nội dung mẫu từ Resource
        String templateContent;
        try (InputStream inputStream = ragPromptTemplate.getInputStream()) {
            templateContent = new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            // Xử lý ngoại lệ
            throw new RuntimeException("Failed to read template file", e);
        }

        String promptText = templateContent
                .replace("$input$", message)
                .replace("$documents$", String.join("\n", contentList));

        OpenAiApi openAiApi = new OpenAiApi(System.getenv("OPENAI_API_KEY"));
        OpenAiChatOptions openAiChatOptions = OpenAiChatOptions.builder()
                .withModel("gpt-4o")
                .withTemperature(0.4F)
                .build();
        OpenAiChatModel chatModel = new OpenAiChatModel(openAiApi, openAiChatOptions);

        ChatResponse chatResponse = ChatClient.builder(chatModel)
                .build().prompt()
                .advisors(new QuestionAnswerAdvisor(vectorStore, SearchRequest.defaults()))
                .user(promptText)
                .call()
                .chatResponse();

        // Lấy nội dung của chatResponse và trả về dưới dạng chuỗi JSON đơn giản
        return chatResponse.getResult();
    }

    private static final String RAG_FILE_PATH = "src/main/resources/docs/rag.txt";

    // API để đọc nội dung từ rag.txt và trả về dưới dạng ResponseEntity
    @GetMapping("/api/get-rag-content")
    public ResponseEntity<String> getRagContent() {
        try {
            String content = new String(Files.readAllBytes(Paths.get(RAG_FILE_PATH)));
            return ResponseEntity.ok(content);
        } catch (IOException e) {
            return ResponseEntity.status(500).body("Không thể đọc nội dung từ file rag.txt");
        }
    }

    // API để xử lý câu hỏi từ người dùng và sử dụng nội dung từ rag.txt để trả lời
    @PostMapping("/api/ask")
    public Generation askQuestion(@RequestBody Map<String, String> request) {
        List<Document> similarDocument = vectorStore.similaritySearch(SearchRequest.query(request.toString()).withTopK(2));
        List<String> contentList = similarDocument.stream().map(Document::getContent).toList();
        String question = request.get("question");

        // Gọi API để lấy nội dung mới nhất từ rag.txt
        String ragContent;
        try {
            ragContent = new String(Files.readAllBytes(Paths.get(RAG_FILE_PATH)), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read template file", e);
        }

        // Đọc nội dung mẫu từ Resource
        String templateContent;
        try (InputStream inputStream = ragPromptTemplate.getInputStream()) {
            templateContent = new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            // Xử lý ngoại lệ
            throw new RuntimeException("Failed to read template file", e);
        }

        // Thay thế các biến trong mẫu
        String prompt = (templateContent
                .replace("$input$", question)
                .replace("$documents$", String.join("\n", contentList))) + ragContent;

        // Gọi API OpenAI để tạo câu trả lời
        OpenAiApi openAiApi = new OpenAiApi(System.getenv("OPENAI_API_KEY"));
        OpenAiChatOptions openAiChatOptions = OpenAiChatOptions.builder()
                .withModel("gpt-4o")
                .withTemperature(0.4F)
                .build();
        OpenAiChatModel chatModel = new OpenAiChatModel(openAiApi, openAiChatOptions);

        ChatResponse chatResponse = ChatClient.builder(chatModel)
                .build().prompt()
                .user(prompt)
                .call()
                .chatResponse();
        return chatResponse.getResult();
    }
}
