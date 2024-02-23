package mlspot.backend.domain.service;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import mlspot.backend.converter.ModelEntityConverter;
import mlspot.backend.data.model.BlogCategoryModel;
import mlspot.backend.data.model.BlogContentModel;
import mlspot.backend.data.model.BlogModel;
import mlspot.backend.data.repository.BlogContentRepository;
import mlspot.backend.data.repository.BlogRepository;
import mlspot.backend.data.repository.BlogCategoryRepository;
import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.domain.entity.BlogCategoryEntity;
import mlspot.backend.domain.entity.BlogCategoryStructureEntity;
import mlspot.backend.exceptions.BlogCategoryNotFoundException;
import mlspot.backend.exceptions.BlogContentNotFoundException;
import mlspot.backend.exceptions.BlogNotFoundException;
import mlspot.backend.presentation.rest.request.CreateBlogCategoryRequest;
import mlspot.backend.presentation.rest.request.CreateBlogContentRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogCategoryRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogContentRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogRequest;

import javax.enterprise.context.ApplicationScoped;
import javax.inject.Inject;
import javax.transaction.Transactional;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@ApplicationScoped
public class BlogService {
    @Inject
    BlogRepository blogRepository;

    @Inject
    BlogContentRepository blogContentRepository;

    @Inject
    BlogCategoryRepository blogCategoryRepository;

    @Transactional
    public List<BlogEntity> getAllBlogs() {
        List<BlogModel> blogModels = blogRepository.findAll().stream().toList();
        List<BlogEntity> blogEntities = new ArrayList<>();
        blogModels.forEach(b -> blogEntities.add(ModelEntityConverter.Of(b)));
        return blogEntities;
    }

    @Transactional
    public BlogEntity getBlog(Long blogId) throws BlogNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        return ModelEntityConverter.Of(blogModel);
    }

    @Transactional
    public BlogEntity createBlog(String title) {
        BlogModel blogModel = new BlogModel().withTitle(title);
        blogModel.persist();
        return ModelEntityConverter.Of(blogModel);
    }

    @Transactional
    public boolean deleteBlog(Long blogId) throws BlogNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        if (!blogRepository.deleteById(blogId))
            return false;
        blogContentRepository
                .findAll()
                .stream()
                .filter(b -> Objects.equals(b.getBlogId(), blogId))
                .forEach(PanacheEntityBase::delete);
        return true;
    }

    @Transactional
    public BlogEntity modifyBlog(ModifyBlogRequest request, Long blogId) throws BlogNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        if (request.getTitle() != null)
            blogModel.setTitle(request.getTitle());
        if (request.getDescription() != null)
            blogModel.setDescription(request.getDescription());
        return ModelEntityConverter.Of(blogModel);
    }

    @Transactional
    public List<BlogContentEntity> getAllBlogContent(Long blogId) throws BlogNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        List<BlogContentModel> blogContentModels = blogContentRepository
                .findAll()
                .stream()
                .filter(b -> b.getBlogId().equals(blogId))
                .toList();
        List<BlogContentEntity> contentEntities = new ArrayList<>();
        blogContentModels.forEach(b -> contentEntities.add(ModelEntityConverter.Of(b)));
        return contentEntities;
    }

    @Transactional
    public BlogContentEntity getBlogContent(Long blogId, Long contentId)
            throws BlogNotFoundException, BlogContentNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        BlogContentModel blogContentModel = blogContentRepository.findById(contentId);
        if (blogContentModel == null)
            throw new BlogContentNotFoundException();
        if (!Objects.equals(blogContentModel.getBlogId(), blogId))
            throw new BlogContentNotFoundException();
        return ModelEntityConverter.Of(blogContentModel);
    }

    @Transactional
    public boolean deleteBlogContent(Long blogId, Long contentId)
            throws BlogNotFoundException, BlogContentNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        BlogContentModel blogContentModel = blogContentRepository.findById(contentId);
        if (blogContentModel == null)
            throw new BlogContentNotFoundException();
        if (!Objects.equals(blogContentModel.getBlogId(), blogId))
            throw new BlogContentNotFoundException();
        Long number = blogContentModel.getNumber();
        blogContentRepository
                .findAll()
                .stream()
                .filter(b -> b.getBlogId().equals(blogId) && b.getNumber() > number)
                .forEach(b -> b.setNumber(b.getNumber() - 1));
        blogContentModel.delete();
        return true;
    }

    @Transactional
    public BlogContentEntity createBlogContent(CreateBlogContentRequest request, Long blogId)
            throws BlogNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        BlogContentModel blogContentModel = new BlogContentModel()
                .withBlogId(blogId)
                .withContent(request.getContent())
                .withType(request.getType())
                .withNumber(blogContentRepository.findAll().stream().filter(b -> b.getBlogId().equals(blogId)).count());
        blogContentModel.persist();
        return ModelEntityConverter.Of(blogContentModel);
    }

    @Transactional
    public BlogContentEntity modifyBlogContent(ModifyBlogContentRequest request, Long blogId, Long contentId)
            throws BlogNotFoundException, BlogContentNotFoundException {
        BlogModel blogModel = blogRepository.findById(blogId);
        if (blogModel == null)
            throw new BlogNotFoundException();
        BlogContentModel blogContentModel = blogContentRepository.findById(contentId);
        if (blogContentModel == null)
            throw new BlogContentNotFoundException();
        if (!Objects.equals(blogContentModel.getBlogId(), blogId))
            throw new BlogContentNotFoundException();
        if (request.getType() != null)
            blogContentModel.setType(request.getType());
        if (request.getContent() != null)
            blogContentModel.setContent(request.getContent());
        if (request.getNumber() != null)
            blogContentModel.setNumber(request.getNumber());
        return ModelEntityConverter.Of(blogContentModel);
    }

    @Transactional
    public BlogCategoryStructureEntity getAllBlogCategory(Long parentId) {
        if (parentId == -1L) {
            List<BlogCategoryStructureEntity> children = blogCategoryRepository
                    .findAll()
                    .stream()
                    .filter(c -> c.getParentId().equals(-1L))
                    .map(c -> getAllBlogCategory(c.getId()))
                    .filter(c -> c != null)
                    .toList();
            BlogCategoryStructureEntity structure = new BlogCategoryStructureEntity()
                    .withName("ROOT")
                    .withId(-1L)
                    .withChildren(children);
            return structure;
        } else {
            BlogCategoryModel blogCategoryModel = blogCategoryRepository.findById(parentId);
            if (blogCategoryModel == null)
                return null;
            List<BlogCategoryStructureEntity> children = blogCategoryRepository
                    .findAll()
                    .stream()
                    .filter(c -> c.getParentId().equals(parentId))
                    .map(c -> getAllBlogCategory(c.getId()))
                    .filter(c -> c != null)
                    .toList();
            BlogCategoryStructureEntity structure = new BlogCategoryStructureEntity()
                    .withName(blogCategoryModel.getName())
                    .withId(parentId)
                    .withChildren(children);
            return structure;
        }
    }

    @Transactional
    public BlogCategoryEntity getBlogCategory(Long categoryId) throws BlogCategoryNotFoundException {
        BlogCategoryModel blogCategoryModel = blogCategoryRepository.findById(categoryId);
        if (blogCategoryModel == null)
            throw new BlogCategoryNotFoundException();
        return ModelEntityConverter.Of(blogCategoryModel);
    }

    @Transactional
    public BlogCategoryEntity createBlogCategory(CreateBlogCategoryRequest request) {
        BlogCategoryModel blogCategoryModel = new BlogCategoryModel().withName(request.getName());
        blogCategoryModel.persist();
        return ModelEntityConverter.Of(blogCategoryModel);
    }

    @Transactional
    public BlogCategoryEntity modifyBlogCategory(ModifyBlogCategoryRequest request, Long categoryId)
            throws BlogCategoryNotFoundException {
        BlogCategoryModel blogCategoryModel = blogCategoryRepository.findById(categoryId);
        if (blogCategoryModel == null)
            throw new BlogCategoryNotFoundException();
        blogCategoryModel.setName(request.getName());
        if (request.getParentId().equals(-1L))
            blogCategoryModel.setParentId(-1L);
        else {
            BlogCategoryModel blogCategoryParentModel = blogCategoryRepository.findById(request.getParentId());
            if (blogCategoryParentModel == null)
                throw new BlogCategoryNotFoundException();
            blogCategoryModel.setParentId(blogCategoryParentModel.getId());
        }
        return ModelEntityConverter.Of(blogCategoryModel);
    }

    @Transactional
    public boolean deleteBlogCategory(Long categoryId) throws BlogCategoryNotFoundException {
        BlogCategoryModel blogCategoryModel = blogCategoryRepository.findById(categoryId);
        if (blogCategoryModel == null)
            throw new BlogCategoryNotFoundException();
        blogCategoryRepository
                .findAll()
                .stream()
                .filter(c -> c.getParentId().equals(categoryId))
                .forEach(c -> c.setParentId(-1L));
        blogCategoryModel.delete();
        return true;
    }
}
