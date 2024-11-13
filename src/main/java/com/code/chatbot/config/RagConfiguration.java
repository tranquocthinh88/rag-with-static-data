package com.code.chatbot.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Configuration
public class RagConfiguration {

    private static final Logger log = LoggerFactory.getLogger(RagConfiguration.class);

    @Value("classpath:/docs/rag.txt")
    private Resource configFile;

    @Value("vectorstore.json")
    private String vectorStoreName;

    @Bean
    SimpleVectorStore simpleVectorStore(EmbeddingModel embeddingModel){
        SimpleVectorStore simpleVectorStore = new SimpleVectorStore(embeddingModel);
        File vectorStoreFile = getVectorStoreFile();
        if(vectorStoreFile.exists()){
            log.info("Vector store file exists");
            simpleVectorStore.load(vectorStoreFile);
        }
        else{
            log.info("Vector store file does not exist, loading...");
            try {
                TextReader textReader = new TextReader(String.valueOf(configFile.getFile()));
                textReader.getCustomMetadata().put("filename", "rag.txt");
                List<Document> documents = textReader.get();
                TokenTextSplitter tokenTextSplitter = new TokenTextSplitter();
                List<Document> splitDocuments = tokenTextSplitter.apply(documents);
                simpleVectorStore.add(splitDocuments);
                simpleVectorStore.save(vectorStoreFile);
            } catch (IOException e) {
                log.error("Error loading text reader: ", e);
            }
        }
        return simpleVectorStore;
    }

    private File getVectorStoreFile(){
        Path path = Paths.get("src", "main", "resources", "data");
        String absolutePath = path.toFile().getAbsolutePath() + "/" + vectorStoreName;
        return new File(absolutePath);
    }
}
