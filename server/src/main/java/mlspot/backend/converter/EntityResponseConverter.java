package mlspot.backend.converter;

import java.util.ArrayList;

import mlspot.backend.domain.entity.BlogCategoryEntity;
import mlspot.backend.domain.entity.BlogCategoryStructureEntity;
import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.domain.entity.ProjectEntity;
import mlspot.backend.presentation.rest.response.BlogContentResponse;
import mlspot.backend.presentation.rest.response.BlogResponse;
import mlspot.backend.presentation.rest.response.ProjectResponse;
import mlspot.backend.presentation.rest.response.BlogCategoryResponse;
import mlspot.backend.presentation.rest.response.BlogCategoryStructureResponse;

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
                .withType(blogContentEntity.getType())
                .withId(blogContentEntity.getId())
                .withNumber(blogContentEntity.getNumber())
                .withBlogId(blogContentEntity.getBlogId());
    }

    public static BlogCategoryResponse Of(BlogCategoryEntity blogCategoryEntity) {
        return new BlogCategoryResponse().withId(blogCategoryEntity.getId())
                .withName(blogCategoryEntity.getName())
                .withParentId(blogCategoryEntity.getParentId());
    }

    public static BlogCategoryStructureResponse Of(BlogCategoryStructureEntity blogCategoryStructureEntity) {
        List<BlogCategoryStructureResponse> children = new ArrayList<BlogCategoryStructureResponse>();
        blogCategoryStructureEntity.getChildren().forEach(e ->  children.add(Of(e)));
        return new BlogCategoryStructureResponse().withId(blogCategoryStructureEntity.getId())
                                                    .withChildren(children)
                                                    .withName(blogCategoryStructureEntity.getName());
    }
}
