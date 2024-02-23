package mlspot.backend.presentation.rest;

import io.vertx.ext.web.RoutingContext;
import mlspot.backend.converter.EntityResponseConverter;
import mlspot.backend.domain.entity.BlogCategoryEntity;
import mlspot.backend.domain.entity.BlogContentEntity;
import mlspot.backend.domain.entity.BlogEntity;
import mlspot.backend.domain.service.BlogService;
import mlspot.backend.errors.BadRequestError;
import mlspot.backend.errors.NotFoundError;
import mlspot.backend.errors.UnauthorizedError;
import mlspot.backend.exceptions.BlogContentNotFoundException;
import mlspot.backend.exceptions.BlogNotFoundException;
import mlspot.backend.exceptions.BlogCategoryNotFoundException;
import mlspot.backend.presentation.rest.request.CreateBlogCategoryRequest;
import mlspot.backend.presentation.rest.request.CreateBlogContentRequest;
import mlspot.backend.presentation.rest.request.CreateBlogRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogContentRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogRequest;
import mlspot.backend.presentation.rest.request.ModifyBlogCategoryRequest;
import mlspot.backend.presentation.rest.response.BlogContentResponse;
import mlspot.backend.presentation.rest.response.BlogResponse;
import mlspot.backend.presentation.rest.response.SuccessResponse;
import mlspot.backend.security.SecurityFilter;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.security.RolesAllowed;
import javax.inject.Inject;
import javax.ws.rs.*;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;

import static mlspot.backend.converter.EntityResponseConverter.Of;

@Path("/blogs")
@Produces(MediaType.APPLICATION_JSON)
public class BlogEndpoint {
    private final Logger logger = LoggerFactory.getLogger(BlogEndpoint.class);

    @Inject
    BlogService blogService;

    @Inject
    SecurityFilter securityFilter;

    @GET
    @Path("/")
    public Response getAllGBlogsEndpoint() {
        logger.info("[GET] /blogs");
        List<BlogEntity> blogEntities = blogService.getAllBlogs();
        List<BlogResponse> blogResponses = new ArrayList<>();
        blogEntities.forEach(b -> blogResponses.add(Of(b)));
        return Response
                .status(200)
                .entity(blogResponses)
                .build();
    }

    @POST
    @Path("/")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response createBlogEndpoint(@RequestBody CreateBlogRequest createBlogRequest, @Context RoutingContext routingContext) {
        logger.info("[POST] /blogs");
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).build();
        if (createBlogRequest == null || createBlogRequest.getTitle() == null)
            return Response.status(400).build();
        BlogEntity blogEntity = blogService.createBlog(createBlogRequest.getTitle());
        return Response.status(200).entity(Of(blogEntity)).build();
    }


    @GET
    @Path("/{blogId}")
    public Response getBlogEndpoint(@PathParam(value = "blogId") Long blogId) {
        logger.info("[GET] /blogs/" + blogId);
        if (blogId == null)
            return Response.status(400).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.getBlog(blogId))).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] could not find blog " + blogId);
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @DELETE
    @Path("/{blogId}")
    public Response deleteBlogEndpoint(@PathParam(value = "blogId") Long blogId, @Context RoutingContext routingContext) {
        logger.info("[DELETE] /blogs/" + blogId);
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).build();
        if (blogId == null)
            return Response.status(400).build();
        try {
            if (blogService.deleteBlog(blogId))
                return Response.status(200).entity(new SuccessResponse()).build();
            return Response.status(400).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] could not find blog " + blogId);
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @PUT
    @RolesAllowed("admin")
    @Path("/{blogId}")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response modifyBlogEndpoint(@RequestBody ModifyBlogRequest request, @PathParam(value = "blogId") Long blogId, @Context RoutingContext routingContext) {
        logger.info("[PUT] /blogs/" + blogId);
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (blogId == null || request == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.modifyBlog(request, blogId))).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] could not find blog " + blogId);
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @GET
    @Path("/{blogId}/content")
    public Response getAllBlogContentEndpoint(@PathParam(value = "blogId") Long blogId) {
        logger.info("[GET] /blogs/" + blogId + "/content");
        if (blogId == null)
            return Response.status(400).entity(new BadRequestException()).build();
        try {
            List<BlogContentEntity> list = blogService.getAllBlogContent(blogId);
            List<BlogContentResponse> response = new ArrayList<>();
            list.forEach(b -> response.add(EntityResponseConverter.Of(b)));
            return Response.status(200).entity(response).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @POST
    @Path("/{blogId}/content")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response createBlogContentEndpoint(@RequestBody CreateBlogContentRequest request, @PathParam(value = "blogId") Long blogId, @Context RoutingContext routingContext) {
        logger.info("[POST] /blogs/" + blogId + "/content");
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (blogId == null || request == null)
            return Response.status(400).entity(new BadRequestException()).build();
        try {
            return Response.status(200).entity(Of(blogService.createBlogContent(request, blogId))).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] blog with id " + blogId + " could not be found");
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @GET
    @Path("/{blogId}/content/{contentId}")
    public Response getBlogContentEndpoint(@PathParam(value = "blogId") Long blogId, @PathParam(value = "contentId") Long contentId) {
        logger.info("[GET] /blogs/" + blogId + "/content/" + contentId);
        if (blogId == null || contentId == null)
            return Response.status(400).entity(new BadRequestException()).build();
        try {
            return Response.status(200).entity(Of(blogService.getBlogContent(blogId, contentId))).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).entity(new NotFoundError()).build();
        } catch (BlogContentNotFoundException ignored) {
            logger.info("[404] blog content with id " + contentId + " could not be found");
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @DELETE
    @Path("/{blogId}/content/{contentId}")
    public Response deleteBlogContentEndpoint(@PathParam(value = "blogId") Long blogId, @PathParam(value = "contentId") Long contentId, @Context RoutingContext routingContext) {
        logger.info("[DELETE] /blogs/" + blogId + "/content/" + contentId);
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (blogId == null || contentId == null)
            return Response.status(400).entity(new BadRequestException()).build();
        try {
            if (blogService.deleteBlogContent(blogId, contentId))
                return Response.status(200).entity(new SuccessResponse()).build();
            return Response.status(400).entity(new BadRequestException()).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).entity(new NotFoundError()).build();
        } catch (BlogContentNotFoundException ignored) {
            logger.info("[404] blog content with id " + contentId + " could not be found");
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @PUT
    @Path("/{blogId}/content/{contentId}")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response modifyBlogContentEndpoint(@RequestBody ModifyBlogContentRequest request, @PathParam(value = "blogId") Long blogId, @PathParam(value = "contentId") Long contentId, @Context RoutingContext routingContext) {
        logger.info("[PUT] /blogs/" + blogId + "/content/" + contentId);
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (blogId == null || contentId == null || request == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.modifyBlogContent(request, blogId, contentId))).build();
        } catch (BlogNotFoundException ignored) {
            logger.info("[404] blog with id " + blogId + " not found");
            return Response.status(404).entity(new NotFoundError()).build();
        } catch (BlogContentNotFoundException ignored) {
            logger.info("[404] blog content with id " + contentId + " not found");
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @GET
    @Path("/category")
    public Response getAllBlogCategoryEndpoint() {
        logger.info("[GET] /blogs/category");
        return Response.status(200).entity(EntityResponseConverter.Of(blogService.getAllBlogCategory(-1L))).build();
    }

    @GET
    @Path("/category/{categoryId}")
    public Response getBlogCategoryEndpoint(@PathParam(value = "categoryId") Long categoryId) {
        logger.info("[GET] /blogs/category/" + categoryId);
        if (categoryId == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(blogService.getBlogCategory(categoryId))).build();
        } catch (BlogCategoryNotFoundException ignored) {
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @POST
    @Path("/category")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response createBlogCategoryEndpoint(@RequestBody CreateBlogCategoryRequest request, @Context RoutingContext context) {
        logger.info("[POST] /blogs/category");
        if (!securityFilter.checkPermission(context))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (request == null || request.getName() == null)
            return Response.status(400).entity(new BadRequestError()).build();
        BlogCategoryEntity blogCategoryEntity = blogService.createBlogCategory(request);
        return Response.status(200).entity(EntityResponseConverter.Of(blogCategoryEntity)).build();
    }

    @PUT
    @Path("/category/{categoryId}")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response modifyBlogCategoryEndpoint(@RequestBody ModifyBlogCategoryRequest request, @PathParam(value = "categoryId") Long categoryId, @Context RoutingContext context) {
        logger.info("[PUT] /blogs/category/" + categoryId);
        if (!securityFilter.checkPermission(context))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (request == null || request.getName() == null || categoryId == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            BlogCategoryEntity blogCategoryEntity = blogService.modifyBlogCategory(request, categoryId);
            return Response.status(200).entity(EntityResponseConverter.Of(blogCategoryEntity)).build();
        } catch (BlogCategoryNotFoundException ignored) {
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @DELETE
    @Path("/category/{categoryId}")
    public Response deleteBlogCategoryEndpoint(@PathParam(value = "categoryId") Long categoryId, @Context RoutingContext context) {
        logger.info("[DELETE] /blogs/category/" + categoryId);
        if (!securityFilter.checkPermission(context))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (categoryId == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            if (!blogService.deleteBlogCategory(categoryId))
                return Response.status(400).entity(new BadRequestError()).build();
            return Response.status(200).entity(new SuccessResponse()).build();
        } catch (BlogCategoryNotFoundException ignored) {
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

}