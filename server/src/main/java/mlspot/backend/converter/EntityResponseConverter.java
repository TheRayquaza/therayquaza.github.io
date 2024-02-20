package mlspot.backend.converter;

import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogContentEnumEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.domain.entity.ProjectEntity;
import mlspot.backend.presentation.rest.response.BlogContentEnumResponse;
import mlspot.backend.presentation.rest.response.BlogContentResponse;
import mlspot.backend.presentation.rest.response.BlogResponse;
import mlspot.backend.presentation.rest.response.ProjectResponse;

import java.util.ArrayList;
import java.util.List;

public class EntityResponseConverter {
    public static ProjectResponse Of(ProjectEntity projectEntity) {
        return new ProjectResponse()
                .withStartingDate(projectEntity.getStartingDate())
                .withFinishedDate(projectEntity.getFinishedData())
                .withId(projectEntity.getId())
                .withName(projectEntity.getName())
                .withLink(projectEntity.getLink())
                .withDescription(projectEntity.getDescription())
                .withMembers(projectEntity.getMembers())
                .withTechnologies(projectEntity.getTechnologies());
    }

    public static BlogResponse Of(BlogEntity blogEntity) {
        return new BlogResponse()
                .withTitle(blogEntity.getTitle())
                .withId(blogEntity.getId())
                .withDescription(blogEntity.getDescription());
    }

    public static BlogContentResponse Of(BlogContentEntity blogContentEntity) {
        return new BlogContentResponse()
                .withContent(blogContentEntity.getContent())
                .withType(Of(blogContentEntity.getBlogContentEnumEntity()))
                .withId(blogContentEntity.getId());
    }

    public static BlogContentEnumResponse Of(BlogContentEnumEntity enumEntity) {
        return switch (enumEntity) {
            case LINK -> BlogContentEnumResponse.LINK;
            case TEXT -> BlogContentEnumResponse.TEXT;
            default -> BlogContentEnumResponse.IMAGE;
        };
    }
}
