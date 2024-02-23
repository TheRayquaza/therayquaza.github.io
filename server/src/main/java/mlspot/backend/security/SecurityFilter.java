package mlspot.backend.security;

import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.RoutingContext;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import javax.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class SecurityFilter {
    @ConfigProperty(name = "api.key")
    String apiKey;

    public boolean checkPermission(RoutingContext request) {
        if (request.request().method() != HttpMethod.GET) {
            String apiKeyHeader = request.request().getHeader("X-api-key");
            return !(apiKeyHeader == null || !apiKeyHeader.equals(apiKey));
        }
        return true;
    }
}
