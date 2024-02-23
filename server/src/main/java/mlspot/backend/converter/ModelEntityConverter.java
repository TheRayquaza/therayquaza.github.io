package mlspot.backend.converter;

import mlspot.backend.data.model.BlogContentModel;
import mlspot.backend.data.model.BlogModel;
import mlspot.backend.data.model.ProjectModel;
import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogContentEnumEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.domain.entity.ProjectEntity;

import java.util.Arrays;

public class ModelEntityConverter {
    public static ProjectEntity Of(ProjectModel projectModel) {
        return new ProjectEntity()
                .withFinishedData(projectModel.getFinishedDate() == null ? "" : projectModel.getFinishedDate().toString())
                .withStartingDate(projectModel.getStartingDate() == null ? "" : projectModel.getStartingDate().toString())
                .withId(projectModel.getId())
                .withName(projectModel.getName())
                .withTechnologies(Arrays.stream(projectModel.getTechnologies().split(",")).toList())
                .withLink(projectModel.getLink())
                .withDescription(projectModel.getDescription())
                .withMembers(projectModel.getMembers());
    }

    public static BlogEntity Of(BlogModel blogModel) {
        return new BlogEntity()
                .withTitle(blogModel.getTitle())
                .withDescription(blogModel.getDescription())
                .withId(blogModel.getId());
    }

    public static BlogContentEntity Of(BlogContentModel blogContentModel) {
        return new BlogContentEntity()
                .withContent(blogContentModel.getContent())
                .withBlogContentEnumEntity(Of(blogContentModel.getType()))
                .withId(blogContentModel.getId())
                .withNumber(blogContentModel.getNumber())
                .withBlogId(blogContentModel.getBlogId());
    }

    public static BlogContentEnumEntity Of(String type) {
        return switch (type) {
            case "LINK" -> BlogContentEnumEntity.LINK;
            case "IMAGE" -> BlogContentEnumEntity.IMAGE;
            default -> BlogContentEnumEntity.TEXT;
        };
    }

}
