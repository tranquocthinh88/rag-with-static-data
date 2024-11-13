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
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

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

//        return ChatClient.builder(chatModel)
//                .build().prompt()
//                .advisors(new QuestionAnswerAdvisor(vectorStore, SearchRequest.defaults()))
//                .user(promptText)
//                .call()
//                .chatResponse();
        ChatResponse chatResponse = ChatClient.builder(chatModel)
                .build().prompt()
                .advisors(new QuestionAnswerAdvisor(vectorStore, SearchRequest.defaults()))
                .user(promptText)
                .call()
                .chatResponse();

        // Lấy nội dung của chatResponse và trả về dưới dạng chuỗi JSON đơn giản
        return chatResponse.getResult();
    }
}
